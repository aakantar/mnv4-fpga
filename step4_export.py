"""
Aşama 2 - Adım 4: Tamsayı parametreleri donanımın okuyacağı ikili dosyalara aktar.

Girdi : mnv4s_int_params.pt (Adım 3)
Çıktı : export/ klasörü
  layers.json  katman tablosu (boyutlar, bit genişlikleri, residual kaynağı, dosya konumları)
  weights.bin  INT8 ağırlıklar, katmanlar art arda, her bölüm 64 bayta hizalı
                 normal / pointwise : [cout][kh][kw][cin]   (OHWI)
                 depthwise          : [kh][kw][c]           (HWC)
  quant.bin    kanal başına 16 bayt, little-endian, her bölüm 64 bayta hizalı:
                 int32  bias        (24 bit aralığında)
                 uint16 m           (15 bit çarpan)
                 uint8  sh          (sağa kaydırma)
                 uint8  reserved
                 uint16 m2_mant     (kısayol çarpanı, residual yoksa 0)
                 uint8  m2_lshift   (kısayol sola kaydırması, residual yoksa 0)
                 uint8  reserved
                 uint32 reserved
  head.bin     ARM tarafı (float32): conv_head.weight [1280][960], conv_head.bias [1280],
               classifier.weight [1000][1280], classifier.bias [1000]

Aktivasyonlar NHWC (kanal son) düzeninde varsayılır.
Katman hesabı:
  acc[c] = sum(w * x) + bias[c]
  y[c]   = clamp((acc*m[c] + skip*(m2_mant[c] << m2_lshift[c]) + 2^(sh[c]-1)) >> sh[c], qmin, qmax)

Kullanım:  python step4_export.py --params mnv4s_int_params.pt --out export
"""
import argparse
import json
import os
import struct

import numpy as np
import torch

ALIGN = 64
QSTRUCT = struct.Struct("<iHBBHBBI")   # 16 bayt
assert QSTRUCT.size == 16


def pad_to(buf, align=ALIGN):
    r = len(buf) % align
    if r:
        buf.extend(b"\0" * (align - r))


def out_hw(h, k, s, p):
    return (h + 2 * p - k) // s + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="mnv4s_int_params.pt")
    ap.add_argument("--out", default="export")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    P = torch.load(args.params, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)
    wbuf, qbuf = bytearray(), bytearray()

    producer = {"input": 0}          # tensör adı -> onu üreten katman numarası (0 = model girişi)
    shapes = {"input": (args.size, args.size, 3)}
    table = []
    for L in P["layers"]:
        idx = L["index"]
        h, w, c = shapes[L["in_tensor"]]
        assert c == L["cin"], (L["conv_name"], c, L["cin"])
        ho, wo = out_hw(h, L["kernel"], L["stride"], L["padding"]), out_hw(w, L["kernel"], L["stride"], L["padding"])
        typ = "depthwise" if L["depthwise"] else ("pointwise" if L["kernel"] == 1 else "dense")

        # Ağırlıklar
        wq = L["w_q"].numpy().astype(np.int8)                     # [cout, cin/g, kh, kw]
        if typ == "depthwise":
            wl = np.ascontiguousarray(wq[:, 0].transpose(1, 2, 0))   # [kh, kw, c]
        else:
            wl = np.ascontiguousarray(wq.transpose(0, 2, 3, 1))      # [cout, kh, kw, cin]
        w_off = len(wbuf)
        wbuf.extend(wl.tobytes())
        w_bytes = wl.nbytes
        pad_to(wbuf)

        # Kanal başına nicemleme parametreleri
        q_off = len(qbuf)
        b, m, sh = L["b_q"].tolist(), L["m"].tolist(), L["sh"].tolist()
        m2m = L["m2_mant"].tolist() if L["m2_mant"] is not None else [0] * L["cout"]
        m2s = L["m2_lshift"].tolist() if L["m2_lshift"] is not None else [0] * L["cout"]
        for ch in range(L["cout"]):
            assert -(1 << 23) <= b[ch] < (1 << 23) and 0 < m[ch] < (1 << 15) and 0 < sh[ch] < 64
            assert 0 <= m2m[ch] < (1 << 15) and 0 <= m2s[ch] < 32
            qbuf.extend(QSTRUCT.pack(int(b[ch]), int(m[ch]), int(sh[ch]), 0, int(m2m[ch]), int(m2s[ch]), 0, 0))
        q_bytes = 16 * L["cout"]
        pad_to(qbuf)

        skip_from = producer[L["skip_tensor"]] if L["skip_tensor"] else None
        table.append(dict(
            index=idx, name=L["conv_name"], block=L["block"], role=L["role"], type=typ,
            cin=L["cin"], cout=L["cout"], kernel=L["kernel"], stride=L["stride"], padding=L["padding"],
            in_shape=[h, w, c], out_shape=[ho, wo, L["cout"]],
            in_from=producer[L["in_tensor"]], skip_from=skip_from,
            in_bits=L["in_bits"], in_signed=not L["in_unsigned"],
            out_bits=L["out_bits"], out_signed=not L["out_unsigned"],
            qmin=L["qmin"], qmax=L["qmax"], relu=bool(L["qmin"] == 0),
            macs=ho * wo * L["cout"] * L["kernel"] ** 2 * (1 if typ == "depthwise" else L["cin"]),
            acc_max_seen=int(L["acc_max_seen"]),
            w_offset=w_off, w_bytes=w_bytes, q_offset=q_off, q_bytes=q_bytes,
            in_tensor=L["in_tensor"], out_tensor=L["out_tensor"], skip_tensor=L["skip_tensor"],
        ))
        producer[L["out_tensor"]] = idx
        shapes[L["out_tensor"]] = (ho, wo, L["cout"])

    with open(os.path.join(args.out, "weights.bin"), "wb") as f:
        f.write(wbuf)
    with open(os.path.join(args.out, "quant.bin"), "wb") as f:
        f.write(qbuf)

    H = P["head"]
    head_parts = [("conv_head.weight", H["conv_head.weight"].reshape(H["conv_head.weight"].shape[0], -1)),
                  ("conv_head.bias", H["conv_head.bias"]),
                  ("classifier.weight", H["classifier.weight"]),
                  ("classifier.bias", H["classifier.bias"])]
    head_meta, off = [], 0
    with open(os.path.join(args.out, "head.bin"), "wb") as f:
        for name, t in head_parts:
            a = t.numpy().astype("<f4")
            f.write(a.tobytes())
            head_meta.append(dict(name=name, shape=list(a.shape), offset=off, bytes=a.nbytes))
            off += a.nbytes

    meta = dict(
        format_version=1,
        description="MobileNetV4-Conv-S tamsayı gövde (FPGA) + float baş (ARM)",
        activation_layout="NHWC",
        weight_layout={"dense": "OHWI [cout][kh][kw][cin]", "pointwise": "OHWI [cout][1][1][cin]",
                       "depthwise": "HWC [kh][kw][c]"},
        quant_struct="<iHBBHBBI: int32 bias, uint16 m, uint8 sh, uint8 res, uint16 m2_mant, "
                     "uint8 m2_lshift, uint8 res, uint32 res",
        requant="y = clamp((acc*m + skip*(m2_mant<<m2_lshift) + (1<<(sh-1))) >> sh, qmin, qmax)",
        alignment=ALIGN,
        input=dict(shape=[args.size, args.size, 3], scale=P["input"]["scale"], qmin=P["input"]["qmin"],
                   qmax=P["input"]["qmax"], mean=list(P["input"]["mean"]), std=list(P["input"]["std"]),
                   preprocess="RGB, kısa kenar 256'ya bicubic, merkezden 224 kırp, /255, (x-mean)/std, "
                              "round(x/scale) ve [qmin,qmax] kırp"),
        output=dict(from_layer=table[-1]["index"], shape=table[-1]["out_shape"], scale=P["output"]["scale"],
                    signed=table[-1]["out_signed"]),
        head=dict(file="head.bin", dtype="float32 little-endian", parts=head_meta,
                  ops="global_avg_pool -> conv_head(1x1)+ReLU -> classifier"),
        results=P["results"],
        totals=dict(layers=len(table), weight_bytes=len(wbuf), quant_bytes=len(qbuf),
                    macs=sum(t["macs"] for t in table)),
        layers=table,
    )
    with open(os.path.join(args.out, "layers.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    print(f"{len(table)} katman | weights.bin {len(wbuf)} B | quant.bin {len(qbuf)} B | "
          f"head.bin {off} B | MAC {meta['totals']['macs'] / 1e6:.1f} M")
    print(f"Yazıldı: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
