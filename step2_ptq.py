"""
Aşama 2 - Adım 2: Kalibrasyon ve INT8 benzetimi (PTQ).

Yaptıkları:
  1) BN'i gömer, gövde ağırlıklarını kanal başına INT8'e nicemler.
  2) Kalibrasyon görüntüleriyle (ImageNet eğitim setinden, sınıf başına 1 = 1000 görüntü)
     her aktivasyon tensörünün dağılımını toplar.
  3) Üç ölçek seçme yöntemini (max, percentile, mse) doğrulama setinin bir alt kümesinde karşılaştırır.
  4) En iyi yöntemle tüm doğrulama setinde W8A8 doğruluğunu ölçer.
  5) Ölçekleri mnv4s_int8_calib.pt dosyasına kaydeder (sonraki adımlar kullanacak).

Kaggle:
  python step2_ptq.py \
    --data  /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val \
    --val-csv /kaggle/input/competitions/imagenet-object-localization-challenge/LOC_val_solution.csv \
    --calib-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train
"""
import argparse
import copy
import os
import time
from types import SimpleNamespace

import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset, Subset

from eval_resolutions import build_dataset, evaluate
from mnv4_quant import (attach_act_quantizers, calibrate, choose_scales, fold_bn,
                        load_model, quantize_weights, set_mode)


class CalibDataset(Dataset):
    """Eğitim setinden her sınıf klasörü için ilk k görüntü (etiket gerekmez)."""

    def __init__(self, train_dir, per_class, transform):
        self.files = []
        for c in sorted(os.listdir(train_dir)):
            d = os.path.join(train_dir, c)
            if os.path.isdir(d):
                imgs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))
                self.files += [os.path.join(d, f) for f in imgs[:per_class]]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB")), 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--labels-from-filename", action="store_true")
    p.add_argument("--calib-dir", default=None,
                   help="ImageNet eğitim klasörü. Verilmezse kalibrasyon doğrulama setinden ayrılır.")
    p.add_argument("--calib-per-class", type=int, default=1)
    p.add_argument("--calib-from-val", type=int, default=1000,
                   help="--calib-dir yoksa doğrulama setinden ayrılacak görüntü sayısı")
    p.add_argument("--weights", default=None)
    p.add_argument("--methods", nargs="+", default=["max", "percentile", "mse"])
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--search-subset", type=int, default=10000,
                   help="Yöntem karşılaştırması için doğrulama alt kümesi boyutu")
    p.add_argument("--relu-unsigned", action="store_true",
                   help="ReLU sonrası aktivasyonları işaretsiz [0,255] nicemle")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--skip-full", action="store_true", help="Tüm set değerlendirmesini atla")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="mnv4s_int8_calib.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    # --- Model: float (BN gömülü) ve nicemlenmiş kopya
    base = load_model(args.weights)
    cfg = base.pretrained_cfg
    folded, _ = fold_bn(base)
    folded = folded.to(device).eval()
    qmodel = copy.deepcopy(folded)
    w_scales = quantize_weights(qmodel, bits=args.bits)
    qs, _ = attach_act_quantizers(qmodel, relu_unsigned=args.relu_unsigned, bits=args.bits)
    print(f"Cihaz: {device} | nicemlenen konvolüsyon: {len(w_scales)} | "
          f"aktivasyon nicemleme noktası: {len(qs)} | bit: {args.bits} | "
          f"ReLU sonrası: {'işaretsiz' if args.relu_unsigned else 'işaretli'}")

    # --- Veri
    tf = create_transform(input_size=(3, 224, 224), is_training=False,
                          interpolation=cfg["interpolation"], mean=cfg["mean"],
                          std=cfg["std"], crop_pct=cfg["crop_pct"])
    val = build_dataset(SimpleNamespace(data=args.data, val_csv=args.val_csv, subset=0,
                                        labels_from_filename=args.labels_from_filename), tf)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(val), generator=g).tolist()
    if args.calib_dir:
        calib = CalibDataset(args.calib_dir, args.calib_per_class, tf)
        eval_idx = perm
        print(f"Kalibrasyon: eğitim setinden {len(calib)} görüntü")
    else:
        calib = Subset(val, perm[:args.calib_from_val])
        eval_idx = perm[args.calib_from_val:]
        print(f"Kalibrasyon: doğrulama setinden ayrılan {len(calib)} görüntü "
              f"(değerlendirme kalan {len(eval_idx)} görüntüde)")
    search = Subset(val, eval_idx[:args.search_subset])
    full = Subset(val, eval_idx)

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(device == "cuda"))

    search_loader = loader(search)

    # --- 1) Referanslar (aynı alt kümede)
    t1_f, _, n = evaluate(folded, search_loader, device)
    print(f"\n[alt küme, {n} görüntü]")
    print(f"  Float (BN gömülü)          : top-1 = {t1_f:.2f}%")
    set_mode(qs, "off")
    t1_w, _, _ = evaluate(qmodel, search_loader, device)
    print(f"  Yalnız ağırlık INT{args.bits} (W8A32) : top-1 = {t1_w:.2f}%  ({t1_w - t1_f:+.2f})")

    # --- 2) Kalibrasyon
    t0 = time.time()
    calibrate(qmodel, qs, loader(calib), device)
    print(f"  Kalibrasyon tamam ({time.time() - t0:.0f} s)")

    # --- 3) Yöntem karşılaştırması
    results = {}
    for m in args.methods:
        scales = choose_scales(qs, m, args.percentile)
        set_mode(qs, "quant")
        t1, t5, _ = evaluate(qmodel, search_loader, device)
        set_mode(qs, "off")
        results[m] = (t1, t5, scales)
        print(f"  W8A8, ölçek yöntemi = {m:10s}: top-1 = {t1:.2f}%  ({t1 - t1_f:+.2f})")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nEn iyi yöntem: {best}")

    # --- 4) En iyi yöntemle tüm set
    best_scales = choose_scales(qs, best, args.percentile)
    if not args.skip_full:
        full_loader = loader(full)
        if args.calib_dir:  # tüm 50.000 görüntü: float sonucu Adım 1'de ölçüldü
            t1_ff, t5_ff = 73.74, 91.43
        else:               # kalibrasyon için ayrılanlar hariç: float'ı da aynı kümede ölç
            t1_ff, t5_ff, _ = evaluate(folded, full_loader, device)
        set_mode(qs, "quant")
        t1_q, t5_q, nq = evaluate(qmodel, full_loader, device)
        set_mode(qs, "off")
        print(f"\n[tüm değerlendirme seti, {nq} görüntü]")
        print(f"  Float : top-1 = {t1_ff:.2f}%  top-5 = {t5_ff:.2f}%")
        print(f"  W8A8  : top-1 = {t1_q:.2f}%  top-5 = {t5_q:.2f}%   (fark {t1_q - t1_ff:+.2f} puan)")
        print(f"  Hedef : top-1 >= 72.74%  ->  {'TUTTU' if t1_q >= 72.74 else 'TUTMADI'}")

    # --- 5) Kaydet
    torch.save(dict(
        method=best, percentile=args.percentile, bits=args.bits, relu_unsigned=args.relu_unsigned,
        act_scales=best_scales,
        act_absmax={n: q.absmax for n, q in qs.items()},
        act_unsigned={n: q.unsigned for n, q in qs.items()},
        weight_scales={k: v.cpu() for k, v in w_scales.items()},
        search_results={k: v[:2] for k, v in results.items()},
    ), args.out)
    print(f"\nKaydedildi: {os.path.abspath(args.out)}  (toplam süre {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
