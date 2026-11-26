"""
MobileNetV4-Conv-S nicemleme kütüphanesi (Aşama 2).
Adım adım genişleyecek. Şu an içinde:
  - load_model : timm modelini yükle (internetten ya da yerel ağırlık dosyasından)
  - fold_bn    : BatchNorm'u konvolüsyon ağırlık ve bias'ına göm
"""
import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"


def load_model(weights=None):
    """Eğitilmiş modeli yükle. weights verilirse internet kullanmadan dosyadan yükler."""
    if weights:
        m = timm.create_model(MODEL_NAME, pretrained=True,
                              pretrained_cfg_overlay=dict(file=weights))
    else:
        m = timm.create_model(MODEL_NAME, pretrained=True)
    return m.eval()


class ActOnly(nn.Module):
    """Gömülmüş BN'in yerine geçer: yalnızca arkasındaki aktivasyonu (ReLU ya da Identity) uygular."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


def _conv_bn_pairs(model):
    """(konvolüsyon, BN, BN'in sahibi modül, BN attribute adı, konvolüsyonun tam adı) listesi."""
    pairs = [(model.conv_stem, model.bn1, model, "bn1", "conv_stem"),
             (model.conv_head, model.norm_head, model, "norm_head", "conv_head")]
    for name, mod in model.named_modules():
        if mod is model:
            continue
        conv = getattr(mod, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            continue
        for bn_attr in ("bn", "bn1"):  # ConvNormAct -> .bn, ConvBnAct -> .bn1
            bn = getattr(mod, bn_attr, None)
            if isinstance(bn, nn.BatchNorm2d):
                pairs.append((conv, bn, mod, bn_attr, f"{name}.conv"))
    return pairs


@torch.no_grad()
def fold_bn(model):
    """
    Her Conv2d + BatchNorm çiftini tek bir Conv2d'ye (bias'lı) dönüştürür.
        y = gamma * (conv(x) + b - mean) / sqrt(var + eps) + beta
          = conv'(x) + b'
        w' = w * gamma / sqrt(var + eps)          (çıkış kanalı başına)
        b' = beta + (b - mean) * gamma / sqrt(var + eps)
    BN'in arkasındaki aktivasyon (ReLU) korunur.
    Orijinal modeli değiştirmez, katlanmış bir kopya döndürür.
    """
    model = copy.deepcopy(model).eval()
    names = []
    for conv, bn, owner, bn_attr, conv_name in _conv_bn_pairs(model):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale.reshape(-1, 1, 1, 1)
        b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        b = bn.bias + (b0 - bn.running_mean) * scale
        conv.weight.copy_(w)
        conv.bias = nn.Parameter(b)
        setattr(owner, bn_attr, ActOnly(bn.act))
        names.append(conv_name)
    remaining = [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    if remaining:
        raise RuntimeError(f"Gömülemeyen BatchNorm kaldı: {remaining}")
    return model, names


# =====================================================================
# Adım 2: Kalibrasyon ve INT8 benzetimi (fake-quant)
#
# Şema (donanım dostu):
#   - Ağırlıklar : kanal başına, simetrik, INT8 [-127, 127]
#   - Aktivasyon : tensör başına, simetrik (sıfır noktası 0)
#                  ReLU sonrası tensörler istenirse işaretsiz [0, 255] (relu_unsigned)
#   - Nicemleme noktaları: model girişi, her gövde konvolüsyonunun çıkışı (ReLU varsa sonrası),
#                  her residual toplamanın çıkışı
#   - Baş kısmı (global pool, conv_head, classifier) ARM'da float kalır
# =====================================================================

QMAX_W = 127


def _body_convs(model):
    """conv_head hariç bütün Conv2d'ler (donanımda çalışacak olanlar)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and n != "conv_head"]


@torch.no_grad()
def quantize_weights(model, bits=8):
    """Gövde konvolüsyonlarının ağırlıklarını kanal başına simetrik nicemle (yerinde).
    Döndürür: {konv_adı: kanal başına ölçek tensörü}"""
    qmax = 2 ** (bits - 1) - 1
    scales = {}
    for name, conv in _body_convs(model):
        w = conv.weight
        s = w.abs().amax(dim=(1, 2, 3)).clamp(min=1e-12) / qmax
        conv.weight.copy_(torch.clamp(torch.round(w / s.view(-1, 1, 1, 1)), -qmax, qmax)
                          * s.view(-1, 1, 1, 1))
        scales[name] = s.clone()
    return scales


def _clip_threshold(M, hist, method, qmax, nbins, percentile=99.99):
    """|x| histogramından (aralık [0, M]) kırpma eşiği T'yi seç."""
    M = M if M > 0 else 1e-8
    if method == "max" or hist is None:
        return M
    h = hist.double().cpu()
    centers = (torch.arange(nbins, dtype=torch.float64) + 0.5) * M / nbins
    if method == "percentile":
        c = torch.cumsum(h, 0) / h.sum().clamp(min=1)
        idx = int(torch.searchsorted(c, torch.tensor(percentile / 100.0, dtype=torch.float64)))
        return float(centers[min(idx, nbins - 1)]) + 0.5 * M / nbins
    if method == "mse":
        # Kırpma + yuvarlama hatasının beklenen karesini histogramdan hesapla, en küçüğünü seç
        best, T = None, M
        for frac in torch.linspace(0.2, 1.0, 161, dtype=torch.float64):
            t = float(frac) * M
            s = t / qmax
            q = torch.clamp(torch.round(centers / s), 0, qmax) * s
            err = float((h * (centers - q) ** 2).sum())
            if best is None or err < best:
                best, T = err, t
        return T
    raise ValueError(method)


class ActQuantizer:
    """Bir aktivasyon tensörü için gözlemci + sahte nicemleyici.
    mode: 'off' (dokunma), 'max' (en büyük |x|'i bul), 'hist' (histogram topla), 'quant' (nicemle)
    per_channel=True: her kanal için ayrı ölçek (yalnızca tüketicisi depthwise olan tensörler için;
    donanımda ölçek, tüketen katmanın kanal başına çarpanına gömülür)."""
    NBINS = 2048

    def __init__(self, name, unsigned=False, bits=8, is_relu=False, per_channel=False):
        self.name, self.bits, self.is_relu = name, bits, is_relu
        self.per_channel = per_channel
        self.set_unsigned(unsigned)
        self.mode = "off"
        self.absmax = None   # tensör başına: float, kanal başına: [C] tensörü
        self.hist = None     # tensör başına: [NBINS], kanal başına: [C, NBINS]
        self.scale = None

    def set_bits(self, bits):
        self.bits = bits
        self.set_unsigned(self.unsigned)

    def set_unsigned(self, unsigned):
        self.unsigned = unsigned
        self.qmin = 0 if unsigned else -(2 ** (self.bits - 1))
        self.qmax = 2 ** self.bits - 1 if unsigned else 2 ** (self.bits - 1) - 1

    def set_per_channel(self, flag):
        """Kalibrasyondan ÖNCE çağrılmalı (istatistikler sıfırlanır)."""
        self.per_channel = flag
        self.absmax, self.hist, self.scale = None, None, None

    def __call__(self, x):
        if self.mode == "max":
            xa = x.detach().abs()
            if self.per_channel:
                m = xa.amax(dim=(0, 2, 3)).float()
                self.absmax = m if self.absmax is None else torch.maximum(self.absmax, m)
            else:
                m = xa.max().item()
                self.absmax = m if self.absmax is None else max(self.absmax, m)
        elif self.mode == "hist":
            xa = x.detach().abs().float()
            if self.per_channel:
                h = torch.stack([torch.histc(xa[:, c], bins=self.NBINS, min=0,
                                             max=max(float(self.absmax[c]), 1e-8))
                                 for c in range(xa.shape[1])])
            else:
                h = torch.histc(xa, bins=self.NBINS, min=0, max=max(self.absmax, 1e-8))
            self.hist = h if self.hist is None else self.hist + h
        elif self.mode == "quant":
            if self.unsigned and x.min() < 0:
                raise RuntimeError(f"{self.name}: işaretsiz seçildi ama negatif değer var")
            s = self.scale.view(1, -1, 1, 1) if self.per_channel else self.scale
            return torch.clamp(torch.round(x / s), self.qmin, self.qmax) * s
        return x

    def choose_scale(self, method, percentile=99.99):
        """Kırpma eşiği T'yi seç, ölçek = T / qmax.
        method: 'max', 'mse', 'percentile' ya da 'p99.9' gibi doğrudan yüzdelik."""
        if method.startswith("p") and method != "percentile":
            method, percentile = "percentile", float(method[1:])
        if self.per_channel:
            T = [_clip_threshold(float(self.absmax[c]),
                                 None if self.hist is None else self.hist[c],
                                 method, self.qmax, self.NBINS, percentile)
                 for c in range(len(self.absmax))]
            self.scale = torch.tensor(T, device=self.absmax.device) / self.qmax
        else:
            M = self.absmax if self.absmax is not None else 0.0
            self.scale = _clip_threshold(M, self.hist, method, self.qmax, self.NBINS, percentile) / self.qmax
        return self.scale


def attach_act_quantizers(model, relu_unsigned=False, bits=8, fuse_add=False):
    """Gömülmüş (fold_bn) modele aktivasyon nicemleyicilerini kancalarla bağla.
    fuse_add=True: residual bloklarda pw_proj çıkışı ayrıca nicemlenmez; kısayol
    32 bit akümülatöre eklenir ve yalnızca toplam nicemlenir (donanımda tek requantization).
    Döndürür: (isim -> ActQuantizer sözlüğü, kanca tutamaçları)"""
    qs, handles = {}, []
    skip_proj = set()
    if fuse_add:
        skip_proj = {f"{n}.pw_proj.bn" for n, m in model.named_modules() if getattr(m, "has_skip", False)}

    # 1) Model girişi (normalize edilmiş görüntü)
    q_in = ActQuantizer("input", unsigned=False, bits=bits)
    qs["input"] = q_in
    handles.append(model.conv_stem.register_forward_pre_hook(lambda m, a: (q_in(a[0]),)))

    # 2) Her gövde konvolüsyonunun çıkışı (ActOnly = BN'in yerine geçen aktivasyon)
    for name, mod in model.named_modules():
        if isinstance(mod, ActOnly) and name != "norm_head" and name not in skip_proj:
            is_relu = isinstance(mod.act, nn.ReLU)
            q = ActQuantizer(name, unsigned=(relu_unsigned and is_relu), bits=bits, is_relu=is_relu)
            qs[name] = q
            handles.append(mod.register_forward_hook(lambda m, a, out, q=q: q(out)))

    # 3) Residual toplamaların çıkışı (blok çıkışı = toplama sonucu)
    for name, mod in model.named_modules():
        if getattr(mod, "has_skip", False):
            q = ActQuantizer(name + ".add", unsigned=False, bits=bits)
            qs[name + ".add"] = q
            handles.append(mod.register_forward_hook(lambda m, a, out, q=q: q(out)))
    return qs, handles


def set_relu_unsigned(quantizers, flag):
    """ReLU çıkışlarını işaretsiz [0,255] (True) ya da işaretli [-128,127] (False) nicemle."""
    for q in quantizers.values():
        if q.is_relu:
            q.set_unsigned(flag)


def set_mode(quantizers, mode):
    for q in quantizers.values():
        q.mode = mode


@torch.no_grad()
def calibrate(model, quantizers, loader, device):
    """İki geçiş: önce |x| maksimumları, sonra histogramlar."""
    model.eval()
    for mode in ("max", "hist"):
        set_mode(quantizers, mode)
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))
    set_mode(quantizers, "off")


def choose_scales(quantizers, method, percentile=99.99):
    return {n: q.choose_scale(method, percentile) for n, q in quantizers.items()}


# =====================================================================
# Adım 3: Yalnızca tamsayı aritmetiğiyle çalışan model (donanımın birebir taklidi)
#
# Her gövde katmanı için:
#   acc[c] = sum(W_q * x_q) + b_q[c]                      (W: INT8, x: UINT8/INT8/INT16, acc: INT32)
#   y[c]   = clamp((acc*m[c] + x_sc*m2[c] + 2^(sh[c]-1)) >> sh[c], qmin, qmax)
#            (x_sc*m2 terimi yalnızca residual bloklarda; ReLU = qmin 0 ile kırpma)
#   m2[c]  = m2_mant[c] << m2_lshift[c]   (m2_mant de MBITS bit: donanımda küçük çarpıcı + kaydırıcı)
#   m: MBITS bitlik pozitif tamsayı, [2^(MBITS-1), 2^MBITS) aralığında
#   >> : aritmetik sağa kaydırma (aşağı yuvarlar) -> +2^(sh-1) ile "yarımı yukarı" yuvarlama
# Konvolüsyonlar float64'te tamsayı değerlerle hesaplanır ve yuvarlanır: |acc| << 2^53 olduğu için kesin.
# =====================================================================

MBITS = 15
BIAS_BITS = 24   # bias (işaretli) bit genişliği üst sınırı


def quantize_multiplier(M, mbits=MBITS):
    """Gerçel çarpan(lar) M -> (m, sh) öyle ki M ~= m * 2^-sh, m in [2^(mbits-1), 2^mbits)."""
    M = torch.as_tensor(M, dtype=torch.float64)
    if (M <= 0).any():
        raise ValueError("çarpan pozitif olmalı")
    e = torch.floor(torch.log2(M))
    sh = (mbits - 1 - e).to(torch.int64)
    m = torch.round(M * torch.pow(2.0, sh.double())).to(torch.int64)
    over = m >= (1 << mbits)            # yuvarlama taşırdıysa bir bit geri al
    m = torch.where(over, m // 2, m)
    sh = torch.where(over, sh - 1, sh)
    if (sh < 1).any():
        raise ValueError("kaydırma 1'den küçük çıktı (çarpan çok büyük)")
    return m, sh


def _qrange(bits, unsigned):
    return (0, 2 ** bits - 1) if unsigned else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


def _load_calib(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class IntModel(nn.Module):
    """Gömülmüş (fold_bn) model + Adım 2 kalibrasyon dosyası -> tamsayı model.
    forward(x_float_normalize) -> logits (gövde tamsayı, baş float).
    self.layers: donanıma aktarılacak katman parametreleri (sırasıyla)."""

    def __init__(self, folded, calib):
        super().__init__()
        self.head_model = folded           # yalnızca forward_head için (ARM tarafı)
        cal = _load_calib(calib) if isinstance(calib, str) else calib
        if not cal.get("fuse_add", False):
            raise ValueError("Bu tamsayı model 'fuse_add=True' ile kalibre edilmiş dosya bekler")
        self.cal = cal
        sc, uns = cal["act_scales"], cal["act_unsigned"]
        bits = cal.get("act_bits", {n: 8 for n in sc})

        def tensor_info(name):
            s = sc[name]
            s = s.double().cpu() if torch.is_tensor(s) else torch.tensor(float(s), dtype=torch.float64)
            qmin, qmax = _qrange(bits[name], uns[name])
            return dict(name=name, scale=s, qmin=qmin, qmax=qmax, bits=bits[name], unsigned=uns[name])

        self.input_info = tensor_info("input")
        layers = []
        cur = self.input_info

        def add_layer(conv, out_name, conv_name, block, role, skip_src=None):
            nonlocal cur
            w = conv.weight.detach().double().cpu()
            b = conv.bias.detach().double().cpu()
            cout = w.shape[0]
            depthwise = conv.groups == conv.in_channels and conv.groups > 1
            s_in = cur["scale"]
            if s_in.numel() > 1 and not depthwise:
                raise NotImplementedError(f"{conv_name}: kanal başına giriş ölçeği yalnızca depthwise için")
            s_w = w.abs().amax(dim=(1, 2, 3)).clamp(min=1e-12) / QMAX_W
            # Alt sınırlar ("ölü" kanallar için): ağırlıkları ~0, bias'ı sıfır olmayan kanallarda
            # ölçek aşırı küçülür ve bias INT32'yi taşar. Ölçeği, bias BIAS_BITS bite sığacak ve
            # katmanın en büyük ölçeğinin 2^-16'sından küçük olmayacak şekilde büyüt.
            s_in_c = s_in if s_in.numel() == 1 else s_in.view(-1)
            s_w_floor = torch.maximum(b.abs() / (s_in_c * (2 ** (BIAS_BITS - 1) - 1)),
                                      s_w.max() * 2.0 ** -16 * torch.ones_like(s_w))
            n_adj = int((s_w < s_w_floor).sum())
            s_w = torch.maximum(s_w, s_w_floor)
            w_q = torch.clamp(torch.round(w / s_w.view(-1, 1, 1, 1)), -QMAX_W, QMAX_W)
            s_acc = s_in_c * s_w
            b_q = torch.round(b / s_acc)
            out = tensor_info(out_name)
            m, sh = quantize_multiplier(s_acc / out["scale"])
            m2 = m2_mant = m2_lshift = None
            if skip_src is not None:
                # Kısayol çarpanı da 15 bitlik çarpan + sola kaydırma: m2 = m2_mant << (sh - sh2)
                m2_mant, sh2 = quantize_multiplier((skip_src["scale"] / out["scale"]).expand(cout))
                m2_lshift = sh - sh2
                if (m2_lshift < 0).any():
                    raise ValueError(f"{conv_name}: kısayol kaydırması negatif")
                m2 = torch.bitwise_left_shift(m2_mant, m2_lshift)
            if b_q.abs().max() >= 2 ** (BIAS_BITS - 1):
                raise OverflowError(f"{conv_name}: bias {BIAS_BITS} bite sığmıyor")
            layers.append(dict(
                index=len(layers) + 1, block=block, role=role, conv_name=conv_name,
                in_tensor=cur["name"], out_tensor=out_name,
                skip_tensor=None if skip_src is None else skip_src["name"],
                cin=conv.in_channels, cout=cout, kernel=conv.kernel_size[0], stride=conv.stride[0],
                padding=conv.padding[0], groups=conv.groups, depthwise=depthwise,
                in_bits=cur["bits"], in_unsigned=cur["unsigned"],
                out_bits=out["bits"], out_unsigned=out["unsigned"], qmin=out["qmin"], qmax=out["qmax"],
                w_q=w_q.to(torch.int8), b_q=b_q.to(torch.int64), m=m, sh=sh, m2=m2,
                m2_mant=m2_mant, m2_lshift=m2_lshift, n_scale_adjusted=n_adj,
                s_in=s_in, s_w=s_w, s_out=out["scale"],
            ))
            cur = out

        # Stem
        add_layer(folded.conv_stem, "bn1", "conv_stem", "stem", "stem")
        for si, stage in enumerate(folded.blocks):
            for bi, blk in enumerate(stage):
                pre = f"blocks.{si}.{bi}"
                if hasattr(blk, "pw_exp"):
                    shortcut = cur
                    for role in ("dw_start", "pw_exp", "dw_mid", "pw_proj"):
                        sub = getattr(blk, role)
                        if isinstance(sub, nn.Identity):
                            continue
                        last = role == "pw_proj"
                        skip = last and blk.has_skip
                        out_name = f"{pre}.add" if skip else f"{pre}.{role}.bn"
                        add_layer(sub.conv, out_name, f"{pre}.{role}.conv", pre, role,
                                  skip_src=shortcut if skip else None)
                else:
                    add_layer(blk.conv, f"{pre}.bn1", f"{pre}.conv", pre, "conv")
        self.layers = layers
        self.out_info = cur

    def quantize_input(self, x):
        s = self.input_info["scale"].to(x.device)
        return torch.clamp(torch.round(x.double() / s), self.input_info["qmin"], self.input_info["qmax"])

    @staticmethod
    def run_layer(L, x, skip=None):
        """x, skip: tamsayı değerli float64 tensörler (cihazda). Çıkış: tamsayı değerli float64."""
        dev = x.device
        acc = F.conv2d(x, L["w_q"].to(dev, torch.float64), None, L["stride"], L["padding"], 1, L["groups"])
        acc = torch.round(acc).to(torch.int64) + L["b_q"].to(dev).view(1, -1, 1, 1)
        v = acc * L["m"].to(dev).view(1, -1, 1, 1)
        if skip is not None:
            v = v + skip.to(torch.int64) * L["m2"].to(dev).view(1, -1, 1, 1)
        sh = L["sh"].to(dev).view(1, -1, 1, 1)
        y = (v + torch.bitwise_left_shift(torch.ones_like(sh), sh - 1)) >> sh
        return torch.clamp(y, L["qmin"], L["qmax"]).double(), acc

    @torch.no_grad()
    def forward_int(self, xq, record_acc=None):
        """Gövdeyi tamsayı olarak çalıştır. record_acc: {katman_no: en büyük |acc|} sözlüğü (isteğe bağlı)."""
        need = {L["skip_tensor"] for L in self.layers if L["skip_tensor"]}  # yalnızca kısayol olarak gerekenleri tut
        tensors = {"input": xq} if "input" in need else {}
        cur = xq
        for L in self.layers:
            skip = tensors.pop(L["skip_tensor"]) if L["skip_tensor"] else None
            cur, acc = self.run_layer(L, cur, skip)
            if L["out_tensor"] in need:
                tensors[L["out_tensor"]] = cur
            if record_acc is not None:
                a = int(acc.abs().max())
                record_acc[L["index"]] = max(record_acc.get(L["index"], 0), a)
        return cur

    @torch.no_grad()
    def forward(self, x):
        q = self.forward_int(self.quantize_input(x))
        feat = (q * self.out_info["scale"].to(q.device)).float()   # ARM: tamsayı -> float
        return self.head_model.forward_head(feat)


def load_fakequant(folded, calib):
    """Kalibrasyon dosyasındaki ölçeklerle fake-quant modeli kur (karşılaştırma için)."""
    cal = _load_calib(calib) if isinstance(calib, str) else calib
    qm = copy.deepcopy(folded)
    quantize_weights(qm)
    qs, _ = attach_act_quantizers(qm, fuse_add=cal.get("fuse_add", False))
    for n, q in qs.items():
        q.per_channel = cal.get("act_per_channel", {}).get(n, False)
        q.set_bits(cal.get("act_bits", {}).get(n, 8))
        q.set_unsigned(cal["act_unsigned"][n])
        s = cal["act_scales"][n]
        q.scale = s.to(next(qm.parameters()).device) if torch.is_tensor(s) else s
        q.mode = "quant"
    return qm, qs
