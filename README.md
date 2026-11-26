# mnv4-fpga

MobileNetV4-Conv-S modelini PYNQ-Z2 (Zynq XC7Z020) üzerinde FPGA ile hızlandırma projesi.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `eval_resolutions.py` | ImageNet doğruluğunu farklı giriş çözünürlüklerinde ölçer |
| `layer_inventory.py` | Katman envanterini (MAC, parametre, bellek) Excel'e çıkarır |
| `mnv4_quant.py` | Nicemleme kütüphanesi (Aşama 2) |
| `step1_fold_check.py` | Aşama 2 / Adım 1: BN folding kontrolü |

## Kaggle'da kullanım

```python
!git clone https://github.com/aakantar/mnv4-fpga.git
%cd mnv4-fpga
```

Sonraki oturumlarda güncel sürümü çekmek için:

```python
%cd /kaggle/working/mnv4-fpga
!git pull
```

ImageNet yolları (Kaggle, "ImageNet Object Localization Challenge" girdisi ekliyken):

```
--data    /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val
--val-csv /kaggle/input/competitions/imagenet-object-localization-challenge/LOC_val_solution.csv
```

## Şimdiye kadarki sonuçlar

- Float32 ImageNet top-1 (224x224): %73.74
- PYNQ-Z2 ARM Cortex-A9 (ncnn, INT8, 2 thread): ~113 ms / görüntü
- PL'den DDR bant genişliği (3 HP port, 100 MHz): ~1.56 GB/s
