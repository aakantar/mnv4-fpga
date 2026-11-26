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


class ActQuantizer:
    """Bir aktivasyon tensörü için gözlemci + sahte nicemleyici.
    mode: 'off' (dokunma), 'max' (en büyük |x|'i bul), 'hist' (histogram topla), 'quant' (nicemle)"""
    NBINS = 2048

    def __init__(self, name, unsigned=False, bits=8):
        self.name, self.unsigned, self.bits = name, unsigned, bits
        self.qmin = 0 if unsigned else -(2 ** (bits - 1))
        self.qmax = 2 ** bits - 1 if unsigned else 2 ** (bits - 1) - 1
        self.mode = "off"
        self.absmax = 0.0
        self.hist = None
        self.scale = None

    def __call__(self, x):
        if self.mode == "max":
            self.absmax = max(self.absmax, x.detach().abs().max().item())
        elif self.mode == "hist":
            h = torch.histc(x.detach().abs().float(), bins=self.NBINS, min=0, max=self.absmax)
            self.hist = h if self.hist is None else self.hist + h
        elif self.mode == "quant":
            if self.unsigned and x.min() < 0:
                raise RuntimeError(f"{self.name}: işaretsiz seçildi ama negatif değer var")
            return torch.clamp(torch.round(x / self.scale), self.qmin, self.qmax) * self.scale
        return x

    def choose_scale(self, method, percentile=99.99):
        """Kırpma eşiği T'yi seç, ölçek = T / qmax."""
        M = self.absmax if self.absmax > 0 else 1e-8
        if method == "max" or self.hist is None:
            T = M
        else:
            h = self.hist.double().cpu()
            centers = (torch.arange(self.NBINS, dtype=torch.float64) + 0.5) * M / self.NBINS
            if method == "percentile":
                c = torch.cumsum(h, 0) / h.sum()
                idx = int(torch.searchsorted(c, torch.tensor(percentile / 100.0, dtype=torch.float64)))
                T = float(centers[min(idx, self.NBINS - 1)]) + 0.5 * M / self.NBINS
            elif method == "mse":
                # Kırpma + yuvarlama hatasının beklenen karesini histogramdan hesapla, en küçüğünü seç
                best, T = None, M
                for frac in torch.linspace(0.2, 1.0, 161, dtype=torch.float64):
                    t = float(frac) * M
                    s = t / self.qmax
                    q = torch.clamp(torch.round(centers / s), 0, self.qmax) * s
                    err = float((h * (centers - q) ** 2).sum())
                    if best is None or err < best:
                        best, T = err, t
            else:
                raise ValueError(method)
        self.scale = T / self.qmax
        return self.scale


def attach_act_quantizers(model, relu_unsigned=False, bits=8):
    """Gömülmüş (fold_bn) modele aktivasyon nicemleyicilerini kancalarla bağla.
    Döndürür: (isim -> ActQuantizer sözlüğü, kanca tutamaçları)"""
    qs, handles = {}, []

    # 1) Model girişi (normalize edilmiş görüntü)
    q_in = ActQuantizer("input", unsigned=False, bits=bits)
    qs["input"] = q_in
    handles.append(model.conv_stem.register_forward_pre_hook(lambda m, a: (q_in(a[0]),)))

    # 2) Her gövde konvolüsyonunun çıkışı (ActOnly = BN'in yerine geçen aktivasyon)
    for name, mod in model.named_modules():
        if isinstance(mod, ActOnly) and name != "norm_head":
            is_relu = isinstance(mod.act, nn.ReLU)
            q = ActQuantizer(name, unsigned=(relu_unsigned and is_relu), bits=bits)
            qs[name] = q
            handles.append(mod.register_forward_hook(lambda m, a, out, q=q: q(out)))

    # 3) Residual toplamaların çıkışı (blok çıkışı = toplama sonucu)
    for name, mod in model.named_modules():
        if getattr(mod, "has_skip", False):
            q = ActQuantizer(name + ".add", unsigned=False, bits=bits)
            qs[name + ".add"] = q
            handles.append(mod.register_forward_hook(lambda m, a, out, q=q: q(out)))
    return qs, handles


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
