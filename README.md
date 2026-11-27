# mnv4-fpga

MobileNetV4-Conv-S modelini PYNQ-Z2 (Zynq XC7Z020) üzerinde FPGA ile hızlandırma projesi.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `eval_resolutions.py` | ImageNet doğruluğunu farklı giriş çözünürlüklerinde ölçer |
| `layer_inventory.py` | Katman envanterini (MAC, parametre, bellek) Excel'e çıkarır |
| `mnv4_quant.py` | Nicemleme kütüphanesi: BN folding, kalibrasyon, fake-quant, tamsayı model |
| `step1_fold_check.py` | Aşama 2 / Adım 1: BN folding kontrolü |
| `step2_ptq.py` | Aşama 2 / Adım 2: ilk PTQ denemesi |
| `step2b_search.py` | Aşama 2 / Adım 2b: ölçek arama (fuse-add, int16, kanal başına) |
| `step3_int.py` | Aşama 2 / Adım 3: yalnızca tamsayı model, doğruluk ve bit genişlikleri |
| `step4_export.py` | Aşama 2 / Adım 4: donanım için ikili dosyalar (export/) |
| `ref_numpy.py` | Yalnızca NumPy ile tamsayı referans model (altın çıktılar) |
| `mnv4s_int8_calib_int16.pt` | Seçilen kalibrasyon (Adım 2b) |
| `mnv4s_int_params.pt` | Tamsayı parametreler (Adım 3) |

## Kaggle'da kullanım

```python
%cd /kaggle/working
!git -C mnv4-fpga pull || git clone https://github.com/aakantar/mnv4-fpga.git
%cd mnv4-fpga
```

ImageNet yolları (Kaggle, "ImageNet Object Localization Challenge" girdisi ekliyken):

```
--data    /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val
--val-csv /kaggle/input/competitions/imagenet-object-localization-challenge/LOC_val_solution.csv
```

## Sonuçlar

| | Top-1 (ImageNet val, 224x224) |
|---|---|
| Float32 | %73.74 |
| Tamsayı model (W8, A8 + bir tensör A16) | %73.05 |

- PYNQ-Z2 ARM Cortex-A9 (ncnn, INT8, 2 thread): ~113 ms / görüntü
- PL'den DDR bant genişliği (3 HP port, 100 MHz): ~1.56 GB/s

## Nicemleme şeması

- Ağırlıklar: kanal başına simetrik INT8
- Aktivasyonlar: tensör başına; ReLU sonrası UINT8, diğerleri INT8; `blocks.2.5.add` INT16
- Residual toplama akümülatörde, tek requantization
- Requantization: `y = clamp((acc*m + skip*(m2_mant<<m2_lshift) + 2^(sh-1)) >> sh, qmin, qmax)`, m 15 bit
- Akümülatör 25 bit işaretli, bias 24 bit işaretli
- Baş (global pool, conv_head, classifier) ARM'da float
