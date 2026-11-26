"""
Aşama 2 - Adım 2b: Aktivasyon ölçeklerini iyileştirme (PTQ analizi ve arama).

Adım 2'de W8A8 doğruluğu hedefin altında kaldı (%71.64, hedef >= %72.74).
Bu script, donanımı değiştirmeden yalnızca ölçek seçimini iyileştirmeye çalışır:
  A) İşaretli / işaretsiz (ReLU sonrası) ve farklı kırpma yöntemlerini karşılaştırır
  B) Her nicemleme noktasının hassasiyetini ölçer (hangisi en çok zarar veriyor)
  C) Her tensör için en iyi kırpma yöntemini tek tek seçer (açgözlü arama)
  D) Sonucu tüm doğrulama setinde (50.000) ölçer ve ölçekleri kaydeder

Arama, ImageNet EĞİTİM setinden ayrı bir kümede yapılır (sınıf başına 2 görüntü = 2000),
böylece doğrulama sonucu aramadan etkilenmez.

Kaggle (GPU önerilir, ~25-30 dk):
  python step2b_search.py \
    --data  .../ILSVRC/Data/CLS-LOC/val --val-csv .../LOC_val_solution.csv \
    --train-dir .../ILSVRC/Data/CLS-LOC/train
"""
import argparse
import copy
import math
import os
import time
from types import SimpleNamespace

import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset

from eval_resolutions import build_dataset, evaluate
from mnv4_quant import (attach_act_quantizers, calibrate, fold_bn, load_model,
                        quantize_weights, set_mode, set_relu_unsigned)

CANDIDATES = ["max", "p99", "p99.9", "p99.99", "p99.999", "mse"]


class TrainPick(Dataset):
    """Eğitim setinde her sınıf klasöründen [start, start+k) aralığındaki görüntüler, etiketli."""

    def __init__(self, train_dir, start, k, transform):
        self.items = []
        classes = sorted(c for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c)))
        for ci, c in enumerate(classes):
            d = os.path.join(train_dir, c)
            imgs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))
            self.items += [(os.path.join(d, f), ci) for f in imgs[start:start + k]]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


@torch.no_grad()
def load_to_device(ds, device, workers, bs=100):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=bs, num_workers=workers):
        xs.append(x.to(device))
        ys.append(y.to(device))
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def forward_all(model, X, bs=250):
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def score(model, X, Y, ref):
    out = forward_all(model, X)
    top1 = 100.0 * (out.argmax(1) == Y).float().mean().item()
    agree = 100.0 * (out.argmax(1) == ref.argmax(1)).float().mean().item()
    sqnr = 10 * math.log10((ref ** 2).sum().item() / max(((out - ref) ** 2).sum().item(), 1e-30))
    return top1, agree, sqnr


def apply_methods(qs, methods):
    """methods: tek bir yöntem adı (hepsine) ya da {tensör: yöntem} sözlüğü."""
    for n, q in qs.items():
        q.choose_scale(methods if isinstance(methods, str) else methods[n])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--labels-from-filename", action="store_true")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--search-per-class", type=int, default=2)
    p.add_argument("--weights", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-full", action="store_true")
    p.add_argument("--out", default="mnv4s_int8_calib.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T0 = time.time()

    base = load_model(args.weights)
    cfg = base.pretrained_cfg
    folded, _ = fold_bn(base)
    folded = folded.to(device).eval()
    qmodel = copy.deepcopy(folded)
    w_scales = quantize_weights(qmodel)
    qs, _ = attach_act_quantizers(qmodel)
    names = list(qs)

    tf = create_transform(input_size=(3, 224, 224), is_training=False,
                          interpolation=cfg["interpolation"], mean=cfg["mean"],
                          std=cfg["std"], crop_pct=cfg["crop_pct"])

    # Kalibrasyon: eğitim setinde sınıf başına 1. görüntü (Adım 2 ile aynı)
    calib = TrainPick(args.train_dir, 0, 1, tf)
    calibrate(qmodel, qs, DataLoader(calib, batch_size=100, num_workers=args.workers), device)
    # Arama kümesi: sınıf başına sonraki görüntüler (kalibrasyondan ayrı)
    X, Y = load_to_device(TrainPick(args.train_dir, 1, args.search_per_class, tf), device, args.workers)
    set_mode(qs, "off")
    ref = forward_all(folded, X)
    print(f"Kalibrasyon {len(calib)} görüntü, arama kümesi {len(X)} görüntü (eğitim setinden) "
          f"| hazırlık {time.time() - T0:.0f} s")
    f_top1 = 100.0 * (ref.argmax(1) == Y).float().mean().item()
    print(f"Float, arama kümesinde top-1 = {f_top1:.2f}%")
    print("(SQNR: nicemli çıktının float çıktıya benzerliği, dB; yüksek = iyi)\n")

    # ---- A) Global yapılandırmalar
    print("A) Global yöntemler")
    print(f"   {'ReLU sonrası':12s} {'yöntem':9s} {'top-1':>7s} {'float ile aynı':>15s} {'SQNR':>7s}")
    best_cfg, best_sqnr = None, -1e9
    for uns in (False, True):
        set_relu_unsigned(qs, uns)
        for m in CANDIDATES:
            apply_methods(qs, m)
            set_mode(qs, "quant")
            t1, ag, sq = score(qmodel, X, Y, ref)
            set_mode(qs, "off")
            print(f"   {'işaretsiz' if uns else 'işaretli':12s} {m:9s} {t1:6.2f}% {ag:14.2f}% {sq:6.2f}")
            if sq > best_sqnr:
                best_cfg, best_sqnr = (uns, m), sq
    uns, gm = best_cfg
    set_relu_unsigned(qs, uns)
    print(f"   En iyi global: ReLU sonrası {'işaretsiz' if uns else 'işaretli'}, yöntem {gm}\n")

    # ---- B) Hassasiyet: diğerleri nicemliyken bu tensör float kalsa SQNR ne kadar artar?
    methods = {n: gm for n in names}
    apply_methods(qs, methods)
    set_mode(qs, "quant")
    _, _, base_sq = score(qmodel, X, Y, ref)
    gains = {}
    for n in names:
        qs[n].mode = "off"
        gains[n] = score(qmodel, X, Y, ref)[2] - base_sq
        qs[n].mode = "quant"
    order = sorted(names, key=lambda n: -gains[n])
    print("B) En hassas 10 nicemleme noktası (yalnızca o float kalsa SQNR kazancı)")
    for n in order[:10]:
        print(f"   {n:32s} +{gains[n]:.2f} dB")
    print()

    # ---- C) Tensör başına açgözlü yöntem seçimi (hassastan başlayarak)
    t0 = time.time()
    cur_sq = base_sq
    changed = 0
    for n in order:
        best_m, best_s = methods[n], cur_sq
        for m in CANDIDATES:
            if m == methods[n]:
                continue
            qs[n].choose_scale(m)
            s = score(qmodel, X, Y, ref)[2]
            if s > best_s + 1e-3:
                best_m, best_s = m, s
        qs[n].choose_scale(best_m)
        if best_m != methods[n]:
            changed += 1
        methods[n], cur_sq = best_m, best_s
    t1, ag, sq = score(qmodel, X, Y, ref)
    set_mode(qs, "off")
    print(f"C) Tensör başına seçim: {changed} tensörün yöntemi değişti ({time.time() - t0:.0f} s)")
    print(f"   arama kümesinde: top-1 = {t1:.2f}%  float ile aynı = {ag:.2f}%  "
          f"SQNR {base_sq:.2f} -> {sq:.2f} dB")
    from collections import Counter
    print(f"   seçilen yöntemlerin dağılımı: {dict(Counter(methods.values()))}\n")

    # ---- D) Doğrulama seti (50.000)
    if not args.skip_full:
        val = build_dataset(SimpleNamespace(data=args.data, val_csv=args.val_csv, subset=0,
                                            labels_from_filename=args.labels_from_filename), tf)
        vl = DataLoader(val, batch_size=128, num_workers=args.workers, pin_memory=(device == "cuda"))
        print("D) Tüm doğrulama seti (50.000 görüntü)")
        apply_methods(qs, gm)
        set_mode(qs, "quant")
        g1, g5, _ = evaluate(qmodel, vl, device)
        apply_methods(qs, methods)
        p1, p5, _ = evaluate(qmodel, vl, device)
        set_mode(qs, "off")
        print(f"   Float                  : top-1 = 73.74%  top-5 = 91.43%")
        print(f"   W8A8, en iyi global    : top-1 = {g1:.2f}%  top-5 = {g5:.2f}%  ({g1 - 73.74:+.2f})")
        print(f"   W8A8, tensör başına    : top-1 = {p1:.2f}%  top-5 = {p5:.2f}%  ({p1 - 73.74:+.2f})")
        print(f"   Hedef top-1 >= 72.74%  ->  {'TUTTU' if p1 >= 72.74 else 'TUTMADI'}")

    apply_methods(qs, methods)
    torch.save(dict(
        method="per-tensor", methods=methods, relu_unsigned=uns, bits=8,
        act_scales={n: q.scale for n, q in qs.items()},
        act_absmax={n: q.absmax for n, q in qs.items()},
        act_unsigned={n: q.unsigned for n, q in qs.items()},
        weight_scales={k: v.cpu() for k, v in w_scales.items()},
        sensitivity_db=gains,
    ), args.out)
    print(f"\nKaydedildi: {os.path.abspath(args.out)}  (toplam {time.time() - T0:.0f} s)")


if __name__ == "__main__":
    main()
