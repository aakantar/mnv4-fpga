"""
Aşama 2 - Adım 1: BN folding kontrolü.

1) Orijinal ve BN'i gömülmüş modelin çıktılarını gerçek görüntülerde karşılaştırır.
2) Gömülmüş modelin 224x224 ImageNet doğruluğunu ölçer (beklenen: ~%73.74, değişmemeli).
3) Gömülmüş ağırlıkları mnv4s_folded.pt olarak kaydeder (sonraki adımlar bunu kullanacak).

Kaggle'da (eval_resolutions.py ve mnv4_quant.py aynı klasörde olmalı):
  python step1_fold_check.py \
    --data /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val \
    --val-csv /kaggle/input/competitions/imagenet-object-localization-challenge/LOC_val_solution.csv
"""
import argparse
import os
from types import SimpleNamespace

import torch
from timm.data import create_transform
from torch.utils.data import DataLoader

from eval_resolutions import build_dataset, evaluate
from mnv4_quant import fold_bn, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--labels-from-filename", action="store_true")
    p.add_argument("--weights", default=None, help="İnternet yoksa yerel model.safetensors")
    p.add_argument("--subset", type=int, default=0, help="0 = tüm 50.000 görüntü")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="mnv4s_folded.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights).to(device)
    folded, names = fold_bn(model)
    folded = folded.to(device)
    print(f"Gömülen Conv+BN çifti: {len(names)}  (beklenen 46: 45 gövde konvolüsyonu + conv_head)")

    cfg = model.pretrained_cfg
    tf = create_transform(input_size=(3, 224, 224), is_training=False,
                          interpolation=cfg["interpolation"], mean=cfg["mean"],
                          std=cfg["std"], crop_pct=cfg["crop_pct"])
    ds_args = SimpleNamespace(data=args.data, val_csv=args.val_csv, subset=args.subset,
                              labels_from_filename=args.labels_from_filename)
    ds = build_dataset(ds_args, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"))

    # 1) Çıktı karşılaştırması: ilk iki batch
    max_abs = max_rel = 0.0
    agree = total = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i == 2:
                break
            x = x.to(device)
            y0, y1 = model(x), folded(x)
            max_abs = max(max_abs, (y0 - y1).abs().max().item())
            max_rel = max(max_rel, ((y0 - y1).abs().max() / y0.abs().max()).item())
            agree += (y0.argmax(1) == y1.argmax(1)).sum().item()
            total += x.shape[0]
    print(f"Çıktı farkı ({total} görüntü): en büyük mutlak = {max_abs:.2e}, "
          f"bağıl = {max_rel:.2e}, top-1 tahmin aynı = {agree}/{total}")

    # 2) Doğruluk
    t1, t5, n = evaluate(folded, loader, device)
    print(f"BN gömülü model, 224x224: top-1 = {t1:.2f}%  top-5 = {t5:.2f}%  ({n} görüntü)")
    print("Referans (orijinal model, 50.000 görüntü): top-1 = 73.74%  top-5 = 91.43%")

    # 3) Kaydet
    torch.save(folded.cpu().state_dict(), args.out)
    print(f"Kaydedildi: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
