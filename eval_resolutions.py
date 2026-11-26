"""
MobileNetV4-Conv-S: farklı giriş çözünürlüklerinde ImageNet doğruluğu ölçümü.

Kullanım örnekleri:
  # ImageNet val, sınıf klasörlerine ayrılmış (val/n01440764/xxx.JPEG ...)
  python eval_resolutions.py --data /yol/imagenet/val

  # Kaggle formatı (tek klasör + LOC_val_solution.csv)
  python eval_resolutions.py --data /yol/ILSVRC/Data/CLS-LOC/val --val-csv /yol/LOC_val_solution.csv

  # Hugging Face formatı (tek klasör, etiket dosya adında: ILSVRC2012_val_00000001_n01751748.JPEG)
  python eval_resolutions.py --data C:/imagenet/val --labels-from-filename

  # İnternetsiz ortam: ağırlığı dosyadan yükle
  python eval_resolutions.py --data ... --weights /yol/model.safetensors

  # Hızlı deneme: sadece 5000 görüntü
  python eval_resolutions.py --data /yol/imagenet/val --subset 5000

Gerekenler: pip install torch torchvision timm pandas
"""
import argparse
import os
import time

import torch
import timm
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from PIL import Image

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"


class KaggleValDataset(Dataset):
    """Kaggle 'imagenet-object-localization-challenge' val formatı:
    tek klasörde ILSVRC2012_val_XXXXXXXX.JPEG dosyaları + etiket CSV'si."""

    def __init__(self, img_dir, csv_path, transform):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["synset"] = df["PredictionString"].str.split().str[0]
        synsets = sorted(df["synset"].unique())  # standart ImageNet sınıf sırası
        self.class_to_idx = {s: i for i, s in enumerate(synsets)}
        self.samples = [
            (os.path.join(img_dir, f"{row.ImageId}.JPEG"), self.class_to_idx[row.synset])
            for row in df.itertuples()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class FilenameLabelDataset(Dataset):
    """Tek klasör; sınıf (synset, ör. n01751748) dosya adının içinde."""

    def __init__(self, img_dir, transform):
        import re
        pat = re.compile(r"(n\d{8})")
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))
        pairs = []
        for f in files:
            m = pat.search(f)
            if m:
                pairs.append((os.path.join(img_dir, f), m.group(1)))
        if not pairs:
            raise RuntimeError(f"{img_dir} içinde adında synset (nXXXXXXXX) geçen görüntü bulunamadı")
        synsets = sorted({s for _, s in pairs})  # standart ImageNet sınıf sırası
        if len(synsets) != 1000:
            print(f"UYARI: {len(synsets)} sınıf bulundu (1000 bekleniyordu), etiket sırası kayabilir")
        self.class_to_idx = {s: i for i, s in enumerate(synsets)}
        self.samples = [(p, self.class_to_idx[s]) for p, s in pairs]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_dataset(args, transform):
    if args.labels_from_filename:
        ds = FilenameLabelDataset(args.data, transform)
    elif args.val_csv:
        ds = KaggleValDataset(args.data, args.val_csv, transform)
    else:
        ds = ImageFolder(args.data, transform=transform)
    if args.subset and args.subset < len(ds):
        # Her çözünürlükte aynı alt küme kullanılsın diye sabit tohum
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[: args.subset].tolist()
        ds = Subset(ds, idx)
    return ds


@torch.no_grad()
def evaluate(model, loader, device):
    top1 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        pred5 = out.topk(5, dim=1).indices
        top1 += (pred5[:, 0] == y).sum().item()
        top5 += (pred5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return 100.0 * top1 / n, 100.0 * top5 / n, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ImageNet val klasörü")
    p.add_argument("--val-csv", default=None, help="Kaggle LOC_val_solution.csv (opsiyonel)")
    p.add_argument("--labels-from-filename", action="store_true",
                   help="Etiketi dosya adından oku (Hugging Face imagenet-1k val formatı)")
    p.add_argument("--weights", default=None,
                   help="İnternet yoksa yerel ağırlık dosyası (Hugging Face'teki model.safetensors)")
    p.add_argument("--resolutions", type=int, nargs="+", default=[224, 160, 128])
    p.add_argument("--subset", type=int, default=0, help="0 = tüm val seti (50.000)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    if args.weights:
        # İnternet yoksa: ağırlık dosyasını (model.safetensors) elle verip yükle
        model = timm.create_model(MODEL_NAME, pretrained=True,
                                  pretrained_cfg_overlay=dict(file=args.weights))
    else:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.eval().to(device)
    cfg = model.pretrained_cfg
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {MODEL_NAME}  ({n_params:.2f}M parametre)\n")

    results = []
    for r in args.resolutions:
        # Eğitimdeki ön işlemenin aynısı: bicubic resize + merkez kırpma (crop_pct=0.875)
        transform = create_transform(
            input_size=(3, r, r),
            is_training=False,
            interpolation=cfg["interpolation"],
            mean=cfg["mean"],
            std=cfg["std"],
            crop_pct=cfg["crop_pct"],
        )
        ds = build_dataset(args, transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == "cuda"))
        t0 = time.time()
        t1, t5, n = evaluate(model, loader, device)
        dt = time.time() - t0
        results.append((r, t1, t5))
        print(f"{r}x{r}: top-1 = {t1:.2f}%  top-5 = {t5:.2f}%  ({n} görüntü, {dt:.0f} s)")

    base = results[0][1]
    print("\nÖzet:")
    print(f"{'Çözünürlük':>11} | {'Top-1':>7} | {'Top-5':>7} | {'Fark (top-1)':>12}")
    for r, t1, t5 in results:
        print(f"{r:>11} | {t1:>6.2f}% | {t5:>6.2f}% | {t1 - base:>+11.2f}")


if __name__ == "__main__":
    main()
