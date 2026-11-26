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
