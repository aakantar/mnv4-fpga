"""
Aşama 2 - Adım 3: Yalnızca tamsayı aritmetiğiyle çalışan model.

Adım 2'nin kalibrasyon dosyasından her katman için INT8 ağırlık, INT32 bias,
çarpan (m) ve kaydırma (sh) üretir; gövdeyi donanımın yapacağı tamsayı işlemleriyle çalıştırır.
  1) ImageNet doğruluğunu ölçer (hedef: top-1 >= %72.74)
  2) Fake-quant modelle tahmin uyumunu ölçer
  3) Donanım için sayı aralıklarını raporlar (akümülatör, kaydırma, bias bit genişlikleri)
  4) Tüm tamsayı parametreleri mnv4s_int_params.pt dosyasına kaydeder (Adım 4 ve Aşama 3 kullanacak)

Kaggle:
  python step3_int.py --calib mnv4s_int8_calib_int16.pt \
    --data .../ILSVRC/Data/CLS-LOC/val --val-csv .../LOC_val_solution.csv
"""
import argparse
import math
import os
import time
from types import SimpleNamespace

import torch
from timm.data import create_transform
from torch.utils.data import DataLoader

from eval_resolutions import build_dataset
from mnv4_quant import IntModel, fold_bn, load_fakequant, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", required=True, help="Adım 2'nin kalibrasyon dosyası (ör. mnv4s_int8_calib_int16.pt)")
    p.add_argument("--data", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--labels-from-filename", action="store_true")
    p.add_argument("--weights", default=None)
    p.add_argument("--subset", type=int, default=0, help="0 = tüm 50.000 görüntü")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="mnv4s_int_params.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T0 = time.time()
    base = load_model(args.weights)
    cfg = base.pretrained_cfg
    folded, _ = fold_bn(base)
    folded = folded.to(device).eval()
    im = IntModel(folded, args.calib).to(device).eval()
    fq, _ = load_fakequant(folded, args.calib)
    fq = fq.to(device).eval()
    L = im.layers
    adj = [(l["conv_name"], l["n_scale_adjusted"]) for l in L if l["n_scale_adjusted"]]
    if adj:
        print("Ağırlık ölçeği alt sınıra çekilen ('ölü') kanallar: " + ", ".join(f"{n}: {k}" for n, k in adj))
    print(f"Cihaz: {device} | tamsayı katman: {len(L)} | residual: {sum(l['skip_tensor'] is not None for l in L)}"
          f" | 16 bit tensör: {[l['out_tensor'] for l in L if l['out_bits'] == 16] or 'yok'}")

    tf = create_transform(input_size=(3, 224, 224), is_training=False,
                          interpolation=cfg["interpolation"], mean=cfg["mean"],
                          std=cfg["std"], crop_pct=cfg["crop_pct"])
    ds = build_dataset(SimpleNamespace(data=args.data, val_csv=args.val_csv, subset=args.subset,
                                       labels_from_filename=args.labels_from_filename), tf)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(device == "cuda"))

    n = c1 = c5 = f1 = agree = 0
    acc_max = {}
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            q = im.forward_int(im.quantize_input(x), acc_max)
            feat = (q * im.out_info["scale"].to(device)).float()
            yi = folded.forward_head(feat)
            yf = fq(x)
            top5 = yi.topk(5, 1).indices
            c1 += (top5[:, 0] == y).sum().item()
            c5 += (top5 == y[:, None]).any(1).sum().item()
            f1 += (yf.argmax(1) == y).sum().item()
            agree += (yi.argmax(1) == yf.argmax(1)).sum().item()
            n += y.numel()
            if i % 100 == 0:
                print(f"  {n} görüntü, tamsayı top-1 şimdilik {100 * c1 / n:.2f}%  ({time.time() - T0:.0f} s)",
                      flush=True)

    t1, t5, ff1 = 100 * c1 / n, 100 * c5 / n, 100 * f1 / n
    print(f"\n[{n} görüntü]")
    print(f"  Float                : top-1 = 73.74%  top-5 = 91.43%   (Adım 1)")
    print(f"  Fake-quant (Adım 2)  : top-1 = {ff1:.2f}%")
    print(f"  TAMSAYI model        : top-1 = {t1:.2f}%  top-5 = {t5:.2f}%   ({t1 - 73.74:+.2f})")
    print(f"  Tamsayı ile fake-quant aynı tahmin: {100 * agree / n:.2f}%")
    print(f"  Hedef top-1 >= 72.74%  ->  {'TUTTU' if t1 >= 72.74 else 'TUTMADI'}")

    # Donanım için sayı aralıkları
    amax = max(acc_max.values())
    sh = torch.cat([l["sh"] for l in L])
    m = torch.cat([l["m"] for l in L])
    bmax = max(int(l["b_q"].abs().max()) for l in L)
    m2m = torch.cat([l["m2_mant"] for l in L if l["m2_mant"] is not None])
    m2s = torch.cat([l["m2_lshift"] for l in L if l["m2_lshift"] is not None])
    print("\nDonanım için sayı aralıkları (görülen en büyük değerler):")
    print(f"  |akümülatör| en büyük = {amax}  -> {math.ceil(math.log2(amax + 1)) + 1} bit işaretli yeterli")
    print(f"  |bias|       en büyük = {bmax}  -> {math.ceil(math.log2(bmax + 1)) + 1} bit işaretli")
    print(f"  çarpan m     : {int(m.min())}..{int(m.max())} (15 bit işaretsiz)")
    print(f"  kaydırma sh  : {int(sh.min())}..{int(sh.max())}")
    print(f"  kısayol çarpanı m2_mant: {int(m2m.min())}..{int(m2m.max())} (15 bit), "
          f"sola kaydırma: {int(m2s.min())}..{int(m2s.max())}")
    top = sorted(acc_max.items(), key=lambda kv: -kv[1])[:5]
    print("  En büyük akümülatörlü 5 katman: " +
          ", ".join(f"{L[k - 1]['conv_name']} ({v})" for k, v in top))

    # Kaydet
    save_layers = []
    for l in L:
        d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in l.items()}
        d["acc_max_seen"] = acc_max.get(l["index"], 0)
        save_layers.append(d)
    torch.save(dict(
        calib_file=os.path.basename(args.calib), mbits=15,
        input=dict(scale=float(im.input_info["scale"]), qmin=im.input_info["qmin"], qmax=im.input_info["qmax"],
                   mean=cfg["mean"], std=cfg["std"]),
        output=dict(tensor=im.out_info["name"], scale=float(im.out_info["scale"])),
        layers=save_layers,
        head={k: v.cpu() for k, v in folded.state_dict().items() if k.startswith(("conv_head", "classifier"))},
        results=dict(int_top1=t1, int_top5=t5, fq_top1=ff1, agree=100 * agree / n, n=n),
    ), args.out)
    print(f"\nKaydedildi: {os.path.abspath(args.out)}  (toplam {time.time() - T0:.0f} s)")


if __name__ == "__main__":
    main()
