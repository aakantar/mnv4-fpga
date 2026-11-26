"""
MobileNetV4-Conv-S katman envanteri.

Her konvolüsyon / FC katmanı için: blok türü, işlem türü, giriş/çıkış boyutu,
kernel, stride, parametre, MAC, INT8 bellek trafiği ve hesap yoğunluğu.
Sonuçları Excel dosyasına (her çözünürlük için bir sayfa + özet) yazar.

Kullanım:
  python layer_inventory.py                      # 224, 160, 128 -> mnv4_conv_s_katmanlar.xlsx
  python layer_inventory.py --resolutions 224 --out tablo.xlsx

Gerekenler: pip install torch timm pandas openpyxl
Hazır ağırlık indirmez; yalnızca mimari yapısını kullanır.
"""
import argparse

import pandas as pd
import timm
import torch
import torch.nn as nn

MODEL_NAME = "mobilenetv4_conv_small"


def uib_type(blk):
    """UIB varyantını iki opsiyonel depthwise'ın varlığına göre belirle."""
    has_start = not isinstance(blk.dw_start, nn.Identity)
    has_mid = not isinstance(blk.dw_mid, nn.Identity)
    return {
        (True, True): "ExtraDW",
        (False, True): "IB",
        (True, False): "ConvNext",
        (False, False): "FFN",
    }[(has_start, has_mid)]


def describe_layers(model):
    """Her Conv2d/Linear için (modül adı -> blok bilgisi) eşlemesi çıkar."""
    info = {}
    info["conv_stem"] = dict(stage="stem", block="Stem", role="stem", residual="")
    for si, stage in enumerate(model.blocks):
        for bi, blk in enumerate(stage):
            prefix = f"blocks.{si}.{bi}"
            bid = f"{si}.{bi}"
            if hasattr(blk, "pw_exp"):  # UniversalInvertedResidual
                t = uib_type(blk)
                skip = "evet" if blk.has_skip else ""
                for role in ["dw_start", "pw_exp", "dw_mid", "pw_proj"]:
                    sub = getattr(blk, role)
                    if isinstance(sub, nn.Identity):
                        continue
                    info[f"{prefix}.{role}.conv"] = dict(
                        stage=si, block=f"{bid} {t}", role=role,
                        residual=skip if role == "pw_proj" else "")
            else:  # ConvBnAct: stage 0-1 FusedIB parçaları, stage 4 son 1x1
                conv = blk.conv
                if si in (0, 1):
                    role = "fused_exp" if conv.kernel_size[0] > 1 else "fused_proj"
                    info[f"{prefix}.conv"] = dict(stage=si, block=f"{si} FusedIB", role=role, residual="")
                else:
                    info[f"{prefix}.conv"] = dict(stage=si, block=f"{bid} Conv1x1", role="conv_1x1", residual="")
    info["conv_head"] = dict(stage="head", block="Head", role="head_1x1 (pool sonrası)", residual="")
    info["classifier"] = dict(stage="head", block="Head", role="fc", residual="")
    return info


def has_act(module_name, model):
    """Katmandan sonra ReLU var mı? (BN'den sonra gelen aktivasyon)"""
    if module_name == "classifier":
        return False
    parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    parent = model.get_submodule(parent_name) if parent_name else model
    if module_name == "conv_stem":
        bn = model.bn1
    elif module_name == "conv_head":
        bn = model.norm_head
    else:
        bn = getattr(parent, "bn", None) or getattr(parent, "bn1", None)
    act = getattr(bn, "act", None)
    return act is not None and not isinstance(act, nn.Identity)


def op_class(mod, name=""):
    if isinstance(mod, nn.Linear) or name == "conv_head":  # conv_head pool sonrası 1x1 uzamsalda: fiilen FC
        return "FC"
    if mod.groups == mod.in_channels and mod.groups > 1:
        return "Depthwise"
    if mod.kernel_size == (1, 1):
        return "Pointwise 1x1"
    return "Dense kxk"


def inventory(res):
    model = timm.create_model(MODEL_NAME, pretrained=False).eval()
    info = describe_layers(model)
    rows = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0]
            if isinstance(mod, nn.Linear):
                cin, cout, k, s, g = mod.in_features, mod.out_features, 1, 1, 1
                hin = win = hout = wout = 1
                macs = cin * cout
            else:
                cin, cout = mod.in_channels, mod.out_channels
                k, s, g = mod.kernel_size[0], mod.stride[0], mod.groups
                hin, win = x.shape[2], x.shape[3]
                hout, wout = out.shape[2], out.shape[3]
                macs = hout * wout * cout * k * k * (cin // g)
            params = mod.weight.numel()
            bias = cout  # BN folding sonrası her çıkış kanalı için bir bias
            act_in = x.numel()
            act_out = out.numel()
            w_bytes = params + 4 * bias  # INT8 ağırlık + INT32 bias
            meta = info[name]
            rows.append(dict(
                sira=len(rows) + 1,
                stage=meta["stage"],
                blok=meta["block"],
                katman=meta["role"],
                islem=op_class(mod, name),
                C_in=cin, C_out=cout, kernel=k, stride=s, groups=g,
                H_in=hin, W_in=win, H_out=hout, W_out=wout,
                relu="evet" if has_act(name, model) else "",
                residual=meta["residual"],
                parametre=params,
                MAC=macs,
                giris_KB=act_in / 1024,
                cikis_KB=act_out / 1024,
                agirlik_KB=w_bytes / 1024,
                MAC_per_bayt=macs / (act_in + act_out + w_bytes),
                modul=name,
            ))
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.register_forward_hook(make_hook(name))
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res))

    df = pd.DataFrame(rows)
    df["MAC_yuzde"] = 100 * df["MAC"] / df["MAC"].sum()
    return df


OPS = ["Dense kxk", "Pointwise 1x1", "Depthwise", "FC"]

# Sayfa sütunları: (başlık, veri anahtarı ya da None=formül)
COLS = ["sira", "stage", "blok", "katman", "islem", "C_in", "C_out", "kernel", "stride", "groups",
        "H_in", "W_in", "H_out", "W_out", "relu", "residual",
        "parametre", "MAC", "giris_KB", "cikis_KB", "agirlik_KB", "MAC_per_bayt", "MAC_yuzde", "modul"]


def write_excel(dfs, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter as L

    font = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)
    head_fill = PatternFill("solid", start_color="D9D9D9")
    col = {name: L(i + 1) for i, name in enumerate(COLS)}

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Ozet"

    for res, df in dfs.items():
        ws = wb.create_sheet(f"{res}x{res}")
        ws.append(COLS)
        n = len(df)
        last = n + 1
        for i, r in enumerate(df.itertuples(index=False), start=2):
            c = col
            row = [r.sira, r.stage, r.blok, r.katman, r.islem, r.C_in, r.C_out, r.kernel, r.stride,
                   r.groups, r.H_in, r.W_in, r.H_out, r.W_out, r.relu, r.residual,
                   f"={c['C_out']}{i}*{c['kernel']}{i}^2*{c['C_in']}{i}/{c['groups']}{i}",
                   f"={c['H_out']}{i}*{c['W_out']}{i}*{c['parametre']}{i}",
                   f"={c['C_in']}{i}*{c['H_in']}{i}*{c['W_in']}{i}/1024",
                   f"={c['C_out']}{i}*{c['H_out']}{i}*{c['W_out']}{i}/1024",
                   f"=({c['parametre']}{i}+4*{c['C_out']}{i})/1024",
                   f"={c['MAC']}{i}/(({c['giris_KB']}{i}+{c['cikis_KB']}{i}+{c['agirlik_KB']}{i})*1024)",
                   f"={c['MAC']}{i}/{c['MAC']}${last + 1}",
                   r.modul]
            ws.append(row)
        # Toplam satırı
        t = last + 1
        ws.cell(t, 1, "TOPLAM")
        for name in ["parametre", "MAC", "giris_KB", "cikis_KB", "agirlik_KB"]:
            ws[f"{col[name]}{t}"] = f"=SUM({col[name]}2:{col[name]}{last})"
        ws[f"{col['MAC_yuzde']}{t}"] = f"=SUM({col['MAC_yuzde']}2:{col['MAC_yuzde']}{last})"
        # Açıklama
        notes = [
            "Not: Satırlar layer_inventory.py ile timm 'mobilenetv4_conv_small' mimarisinden üretildi (Conv2d ve Linear katmanları, çalışma sırasıyla).",
            "Varsayım: INT8 ağırlık/aktivasyon, BN ağırlıklara gömülü, kanal başına INT32 bias. KB = 1024 bayt.",
            "Global average pooling 45. ve 46. satır arasında; residual toplama 'residual=evet' olan katmanın çıkışında yapılır.",
        ]
        for k, txt in enumerate(notes):
            ws.cell(t + 2 + k, 1, txt)

        # Biçim
        for row in ws.iter_rows():
            for cell in row:
                cell.font = font
        for cell in ws[1]:
            cell.font = bold
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center")
        for name in COLS:
            ws.cell(t, COLS.index(name) + 1).font = bold
        for i in range(2, t + 1):
            for name in ["parametre", "MAC"]:
                ws[f"{col[name]}{i}"].number_format = "#,##0"
            for name in ["giris_KB", "cikis_KB", "agirlik_KB", "MAC_per_bayt"]:
                ws[f"{col[name]}{i}"].number_format = "0.0"
            ws[f"{col['MAC_yuzde']}{i}"].number_format = "0.0%"
        widths = {"blok": 14, "katman": 22, "islem": 15, "modul": 26, "MAC": 13, "parametre": 11}
        for name in COLS:
            ws.column_dimensions[col[name]].width = widths.get(name, max(len(name) + 2, 8))
        ws.freeze_panes = "A2"

    # Özet sayfası: işlem türüne göre SUMIFS/COUNTIF formülleri
    ws = ws_sum
    hdr = ["cozunurluk", "islem", "katman_sayisi", "MAC", "MAC_yuzde", "parametre", "param_yuzde"]
    ws.append(hdr)
    r = 2
    for res, df in dfs.items():
        sh = f"'{res}x{res}'"
        n_last = len(df) + 1
        rng = lambda name: f"{sh}!${col[name]}$2:${col[name]}${n_last}"
        first = r
        for op in OPS:
            ws.append([f"{res}x{res}", op,
                       f'=COUNTIF({rng("islem")},B{r})',
                       f'=SUMIFS({rng("MAC")},{rng("islem")},B{r})',
                       f"=D{r}/D${first + len(OPS)}",
                       f'=SUMIFS({rng("parametre")},{rng("islem")},B{r})',
                       f"=F{r}/F${first + len(OPS)}"])
            r += 1
        ws.append([f"{res}x{res}", "TOPLAM", f"=SUM(C{first}:C{r - 1})", f"=SUM(D{first}:D{r - 1})",
                   f"=SUM(E{first}:E{r - 1})", f"=SUM(F{first}:F{r - 1})", f"=SUM(G{first}:G{r - 1})"])
        r += 1
        ws.append([])
        r += 1
    ws.cell(r + 1, 1, "Not: Değerler çözünürlük sayfalarındaki katman tablolarından formülle hesaplanır.")
    for row in ws.iter_rows():
        for cell in row:
            cell.font = font
            if cell.row > 1 and cell.column in (4, 6):
                cell.number_format = "#,##0"
            if cell.row > 1 and cell.column in (5, 7):
                cell.number_format = "0.0%"
            if cell.value == "TOPLAM":
                for c in ws[cell.row]:
                    c.font = bold
    for cell in ws[1]:
        cell.font = bold
        cell.fill = head_fill
    for letter, w in zip("ABCDEFG", [12, 15, 14, 14, 11, 12, 12]):
        ws.column_dimensions[letter].width = w
    ws.freeze_panes = "A2"
    wb.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolutions", type=int, nargs="+", default=[224, 160, 128])
    p.add_argument("--out", default="mnv4_conv_s_katmanlar.xlsx")
    args = p.parse_args()

    dfs = {r: inventory(r) for r in args.resolutions}
    write_excel(dfs, args.out)

    for r, df in dfs.items():
        print(f"\n=== {r}x{r}: {len(df)} katman, {df['MAC'].sum()/1e6:.1f} M MAC, "
              f"{df['parametre'].sum()/1e6:.2f} M parametre ===")
        g = df.groupby("islem").agg(katman=("MAC", "size"), MAC=("MAC", "sum"), parametre=("parametre", "sum"))
        g["MAC_%"] = 100 * g["MAC"] / g["MAC"].sum()
        g["param_%"] = 100 * g["parametre"] / g["parametre"].sum()
        print(g.reindex(OPS).to_string(float_format=lambda v: f"{v:.1f}"))
    print(f"\nKaydedildi: {args.out}")


if __name__ == "__main__":
    main()
