"""
MobileNetV4-Conv-S tamsayı referans modeli (yalnızca NumPy).

Yalnızca export/ klasöründeki dosyaları okur (layers.json, weights.bin, quant.bin, head.bin);
PyTorch ya da timm gerektirmez. FPGA'nın üretmesi gereken "altın" çıktıları üretir.

Kullanım:
  python ref_numpy.py --export export --image kedi.jpg            # sınıflandır
  python ref_numpy.py --export export --image kedi.jpg --dump golden   # katman çıktılarını kaydet
"""
import argparse
import json
import os

import numpy as np

QDTYPE = np.dtype([("bias", "<i4"), ("m", "<u2"), ("sh", "u1"), ("r0", "u1"),
                   ("m2_mant", "<u2"), ("m2_lshift", "u1"), ("r1", "u1"), ("r2", "<u4")])


class RefModel:
    def __init__(self, export_dir):
        with open(os.path.join(export_dir, "layers.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.layers = self.meta["layers"]
        wraw = np.fromfile(os.path.join(export_dir, "weights.bin"), dtype=np.int8)
        qraw = np.fromfile(os.path.join(export_dir, "quant.bin"), dtype=QDTYPE)
        hraw = np.fromfile(os.path.join(export_dir, "head.bin"), dtype="<f4")
        for t in self.layers:
            w = wraw[t["w_offset"]: t["w_offset"] + t["w_bytes"]].astype(np.int64)
            k = t["kernel"]
            t["_w"] = w.reshape(k, k, t["cout"]) if t["type"] == "depthwise" else w.reshape(t["cout"], k, k, t["cin"])
            q = qraw[t["q_offset"] // 16: t["q_offset"] // 16 + t["cout"]]
            t["_bias"] = q["bias"].astype(np.int64)
            t["_m"] = q["m"].astype(np.int64)
            t["_sh"] = q["sh"].astype(np.int64)
            t["_m2"] = np.left_shift(q["m2_mant"].astype(np.int64), q["m2_lshift"].astype(np.int64))
        self.head = {}
        for p in self.meta["head"]["parts"]:
            n = p["bytes"] // 4
            self.head[p["name"]] = hraw[p["offset"] // 4: p["offset"] // 4 + n].reshape(p["shape"]).astype(np.float64)

    # ---------------- Ön işleme (ARM) ----------------
    def preprocess(self, img_path):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        size = self.meta["input"]["shape"][0]
        short = int(round(size / 0.875))                       # 256
        w, h = img.size
        if w <= h:
            nw, nh = short, int(short * h / w)
        else:
            nw, nh = int(short * w / h), short
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
        img = img.crop((left, top, left + size, top + size))
        x = np.asarray(img, dtype=np.float32) / 255.0                  # HWC
        x = (x - np.array(self.meta["input"]["mean"], np.float32)) / np.array(self.meta["input"]["std"], np.float32)
        return self.quantize_input(x)

    def quantize_input(self, x_float_hwc):
        s = self.meta["input"]["scale"]
        q = np.round(x_float_hwc.astype(np.float64) / s)
        return np.clip(q, self.meta["input"]["qmin"], self.meta["input"]["qmax"]).astype(np.int64)

    # ---------------- Tamsayı katman (FPGA'nın yapacağı) ----------------
    @staticmethod
    def conv_acc(t, x):
        """x: [H, W, C] int64 -> akümülatör [Ho, Wo, cout] int64 (bias hariç)."""
        k, s, p = t["kernel"], t["stride"], t["padding"]
        ho, wo, _ = t["out_shape"]
        xp = np.pad(x, ((p, p), (p, p), (0, 0)))                    # sıfır dolgu (sıfır noktası 0)
        if t["type"] == "depthwise":
            acc = np.zeros((ho, wo, t["cout"]), np.int64)
            for i in range(k):
                for j in range(k):
                    acc += t["_w"][i, j] * xp[i: i + s * ho: s, j: j + s * wo: s, :]
            return acc
        # im2col: [Ho, Wo, k, k, C] -> matris çarpımı (float64'te kesin: tüm değerler < 2^53)
        sH, sW, sC = xp.strides
        patches = np.lib.stride_tricks.as_strided(
            xp, shape=(ho, wo, k, k, xp.shape[2]), strides=(sH * s, sW * s, sH, sW, sC))
        A = patches.reshape(ho * wo, -1).astype(np.float64)
        B = t["_w"].reshape(t["cout"], -1).astype(np.float64)
        return np.rint(A @ B.T).astype(np.int64).reshape(ho, wo, t["cout"])

    @classmethod
    def run_layer(cls, t, x, skip=None):
        acc = cls.conv_acc(t, x) + t["_bias"]
        v = acc * t["_m"]
        if skip is not None:
            v = v + skip * t["_m2"]
        sh = t["_sh"]
        y = (v + np.left_shift(np.int64(1), sh - 1)) >> sh
        return np.clip(y, t["qmin"], t["qmax"]), acc

    def forward_body(self, xq):
        outs = {0: xq}
        for t in self.layers:
            skip = outs[t["skip_from"]] if t["skip_from"] is not None else None
            outs[t["index"]], _ = self.run_layer(t, outs[t["in_from"]], skip)
        return outs

    # ---------------- Baş (ARM, float) ----------------
    def head_forward(self, q_last):
        f = q_last.astype(np.float64) * self.meta["output"]["scale"]   # [7, 7, 960]
        f = f.mean(axis=(0, 1))                                          # global avg pool
        h = self.head["conv_head.weight"] @ f + self.head["conv_head.bias"]
        h = np.maximum(h, 0)                                             # ReLU
        return self.head["classifier.weight"] @ h + self.head["classifier.bias"]

    def __call__(self, xq):
        outs = self.forward_body(xq)
        return self.head_forward(outs[self.layers[-1]["index"]]), outs


def class_names():
    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        return [info.index_to_description(i) for i in range(1000)]
    except Exception:
        return [f"sınıf {i}" for i in range(1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="export")
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump", default=None, help="Katman çıktılarını bu klasöre .npy olarak kaydet")
    args = ap.parse_args()

    ref = RefModel(args.export)
    xq = ref.preprocess(args.image)
    logits, outs = ref(xq)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    names = class_names()
    print(f"{os.path.basename(args.image)}: ilk 5 tahmin (tamsayı gövde + float baş)")
    for i in np.argsort(-p)[:5]:
        print(f"  {p[i] * 100:6.2f}%  [{i:3d}] {names[i]}")
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        np.save(os.path.join(args.dump, "input_q.npy"), xq.astype(np.int16))
        for t in ref.layers:
            dt = np.int16 if t["out_bits"] == 16 else (np.int8 if t["out_signed"] else np.uint8)
            np.save(os.path.join(args.dump, f"layer{t['index']:02d}.npy"), outs[t["index"]].astype(dt))
        np.save(os.path.join(args.dump, "logits.npy"), logits.astype(np.float32))
        print(f"Altın çıktılar kaydedildi: {os.path.abspath(args.dump)}")


if __name__ == "__main__":
    main()
