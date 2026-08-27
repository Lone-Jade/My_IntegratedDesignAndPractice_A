#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体版打印稿生成器（正面字卡 + 背面编号，双面打印）
=====================================================
与 tools/make_print_sheet.py 同款排版/输出格式，但字符来源是系统字体
而不是 MNIST 手写体，用于扩充"打印字符"数据集：
  * 每个数字 0-9 用 10 种字体各渲染 1 张（默认字体列表见 DEFAULT_FONTS，
    可用 --fonts 覆盖成任意字体文件）；
  * 背面编号默认从 x-21 开始（--start-seq 21），即
    0-21..0-30、1-21..1-30、...、9-21..9-30，与现有扫描数据集
    datasets/N_scan/N-01..N-20 的编号无缝衔接；
  * 输出与 make_print_sheet.py 完全一致：正/背面 300dpi A4 PNG +
    单个 PDF（页序 正1背1正2背2...）+ 图例 txt。

字符渲染（与 MNIST 版风格统一）：
  字体字形 -> 2x 超采样光栅化 -> 墨迹包围盒裁剪 -> 等比放大填满
  5x5cm 方块并居中；默认做"合成加粗"（--thicken auto，每侧外扩约
  4.5% 字符宽度），使打印笔画宽度与 MNIST 版（加粗后约 0.7cm）相当，
  保证 8 路 TCRT5000（间距约 0.65cm）扫描不漏检。关掉加粗用
  --thicken 0，改粗细用 --thicken N（N=每侧外扩像素，300dpi 下
  1px ≈ 0.085mm）。

用法:
    python3 tools/make_print_sheet_fonts.py                      # 默认10字体, x-21..x-30
    python3 tools/make_print_sheet_fonts.py --fonts a.ttf,b.ttf  # 自定义字体(逗号分隔)
    python3 tools/make_print_sheet_fonts.py --thicken 0          # 不加粗
    python3 tools/make_print_sheet_fonts.py --start-seq 31       # 编号从 x-31 开始
    python3 tools/make_print_sheet_fonts.py --out-dir ./print-fonts

输出:
    print-fonts/sheet-1-front.png, sheet-1-back.png ...   (300dpi A4)
    print-fonts/print_sheets.pdf                          (页序 正1背1正2背2...)
    print-fonts/sheet-1.txt ...                           (行列位置 = 数字-序号 + 字体名)

依赖: numpy, Pillow（PIL）
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 让 "import make_print_sheet" 在任意 cwd 下都能找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_print_sheet as mps  # 复用排版/PNG/PDF 逻辑: A4_PX, draw_dashed_rect,
                                # _grid_margins, compose_back_sheet, save_gray_png, write_pdf

# 默认 10 种字体（均为系统自带、支持拉丁数字的字体，风格尽量拉开）：
#   1 DejaVu Sans          2 DejaVu Serif          3 DejaVu Sans Mono
#   4 Liberation Sans      5 Liberation Serif      6 FreeSerif
#   7 FreeMono             8 URW Gothic            9 C059 (Century Schoolbook)
#   10 Z003 (Zapf Chancery 手写风)
# 某路径不存在时自动跳过，仍会取足 10 个可用字体；也可 --fonts 整体替换。
DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
]
# 备选字体（默认列表缺文件时按此顺序补足）
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Demi.otf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Kinnari.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/Umpush.ttf",
    "/usr/share/fonts/truetype/tlwg/Waree.ttf",
    "/usr/share/fonts/truetype/tlwg/Purisa.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]


def font_display_name(path):
    """返回 (family, style)，用于图例展示。"""
    try:
        family, style = ImageFont.truetype(path, 12).getname()
        return family, style
    except Exception:
        return os.path.basename(path), ""


def resolve_fonts(user_fonts, want=10):
    """确定实际使用的字体文件列表。

    user_fonts: 非空则用逗号分隔的路径列表（原样全部使用）；
    否则用 DEFAULT_FONTS + FALLBACK_FONTS，取前 want 个存在且能加载的。
    返回 [(path, family, style), ...]。
    """
    if user_fonts:
        cands = [p.strip() for p in user_fonts.split(",") if p.strip()]
    else:
        cands = DEFAULT_FONTS + FALLBACK_FONTS
    fonts = []
    for p in cands:
        if not os.path.isfile(p):
            print(f"  跳过: 字体文件不存在 {p}")
            continue
        try:
            ImageFont.truetype(p, 12)
        except Exception as e:
            print(f"  跳过: 字体无法加载 {p} ({e})")
            continue
        family, style = font_display_name(p)
        fonts.append((p, family, style))
        if not user_fonts and len(fonts) >= want:
            break
    return fonts


def render_digit_text(digit, font_path, digit_px, thicken=0, ss=2, pad_frac=0.02):
    """用 TrueType 字体把单个数字渲染成 digit_px×digit_px 的 0/255 图（0=墨）。

    流程: 2x 超采样光栅化 -> 合成加粗(可选) -> 墨迹包围盒裁剪(带 2% 边)
          -> 等比放大填满方块并居中，边沿抗锯齿。
    thicken: 每侧外扩像素数（在最终分辨率下），0 表示不加粗。
    返回 (img uint8 (digit_px,digit_px), 可见墨像素数)；渲染失败返回 (None, 0)。
    """
    canvas = digit_px * ss
    font_size = int(canvas * 0.6)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None, 0
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    cx, cy = canvas // 2, canvas // 2
    if thicken > 0:
        r = max(1, round(thicken * ss))
        offs = [(dx, dy) for dx in (-r, 0, r) for dy in (-r, 0, r)]  # 3x3 合成加粗
    else:
        offs = [(0, 0)]
    for dx, dy in offs:
        d.text((cx + dx, cy + dy), str(digit), font=font, fill=0, anchor="mm")

    a = np.array(img)
    ink = a < 128
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None, 0
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1   # 未加边包围盒
    pad = int(round(canvas * pad_frac))
    x0, x1 = max(0, xs.min() - pad), min(canvas, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(canvas, ys.max() + 1 + pad)
    crop = img.crop((x0, y0, x1, y1))
    s = digit_px / max(bh, bw)                                   # 墨迹按包围盒填满方块
    nw, nh = max(1, round(crop.width * s)), max(1, round(crop.height * s))
    crop = crop.resize((nw, nh), Image.LANCZOS)                  # 留白随之缩放，可能超出方块
    arr = np.asarray(crop, np.float64)
    if arr.shape[0] > digit_px:                                  # 取中心区域(墨迹对称在中间)
        oy = (arr.shape[0] - digit_px) // 2
        arr = arr[oy:oy + digit_px, :]
    if arr.shape[1] > digit_px:
        ox = (arr.shape[1] - digit_px) // 2
        arr = arr[:, ox:ox + digit_px]
    out = np.full((digit_px, digit_px), 255.0, np.float64)
    oy, ox = (digit_px - arr.shape[0]) // 2, (digit_px - arr.shape[1]) // 2
    out[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return out.astype(np.uint8), int(ink.sum())


def compose_sheet_fonts(cells, card_w_px, card_h_px, digit_px, cols=2, gap=40, frame=True):
    """cells: [(label, seq, font_name, digit_img(digit_px×digit_px))] -> 正面 A4 画布。

    与 make_print_sheet.compose_sheet 同款排版：每行 cols 个卡片（默认2）、
    网格居中、字符方块居中、可选虚线裁切框。
    """
    H, W = mps.A4_PX
    rows = (len(cells) + cols - 1) // cols
    mx, my = mps._grid_margins(card_w_px, card_h_px, cols, rows, gap)
    canvas = np.full((H, W), 255, np.uint8)
    off_x = (card_w_px - digit_px) // 2
    off_y = (card_h_px - digit_px) // 2
    for i, (_, _, _, img) in enumerate(cells):
        r, c = divmod(i, cols)
        x0 = mx + c * (card_w_px + gap)
        y0 = my + r * (card_h_px + gap)
        if x0 + card_w_px > W or y0 + card_h_px > H:
            break
        card = np.full((card_h_px, card_w_px), 255, np.uint8)
        card[off_y:off_y + digit_px, off_x:off_x + digit_px] = img
        if frame:
            mps.draw_dashed_rect(card, 0, 0, card_w_px, card_h_px, inset=0)
        canvas[y0:y0 + card_h_px, x0:x0 + card_w_px] = card
    return canvas


def main():
    ap = argparse.ArgumentParser(
        description="字体版打印稿生成器（正面字卡+背面编号，编号从 x-21 开始，双面打印）")
    ap.add_argument("--fonts", default="",
                    help="逗号分隔的字体文件路径，覆盖默认10字体（默认用系统自带字体列表）")
    ap.add_argument("--start-seq", type=int, default=21,
                    help="每个数字的起始序号(默认21 -> x-21..x-30)")
    ap.add_argument("--thicken", default="auto",
                    help="合成加粗每侧像素数: auto=按字符宽度4.5%自动(默认, 约27px), "
                         "0=不加粗, N=指定像素(300dpi下1px≈0.085mm)")
    ap.add_argument("--digit-cm", type=float, default=5.0, help="字符最大尺寸 cm(默认5)")
    ap.add_argument("--card-w-cm", type=float, default=9.0, help="虚线框宽 cm(默认9)")
    ap.add_argument("--card-h-cm", type=float, default=6.0, help="虚线框高 cm(默认6)")
    ap.add_argument("--cols", type=int, default=2, help="每行卡片个数(默认2)")
    ap.add_argument("--no-frame", action="store_true", help="不打印卡片虚线裁切框")
    ap.add_argument("--back-flip", choices=["long", "short", "none"], default="none",
                    help="背面镜像方式: none=自动双面驱动代偿(默认), "
                         "long=手动双面翻长边, short=手动双面翻短边")
    ap.add_argument("--back-text-scale", type=int, default=14, help="背面编号字号(默认14)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", default="./print-fonts", help="输出目录(默认 ./print-fonts)")
    args = ap.parse_args()

    digit_px = int(round(args.digit_cm / 2.54 * args.dpi))
    card_w_px = int(round(args.card_w_cm / 2.54 * args.dpi))
    card_h_px = int(round(args.card_h_cm / 2.54 * args.dpi))
    if digit_px > card_w_px - 20 or digit_px > card_h_px - 20:
        sys.exit(f"字符 {args.digit_cm}cm 大于卡片 {args.card_w_cm}x{args.card_h_cm}cm，"
                 "请减小 --digit-cm 或增大 --card-w-cm/--card-h-cm")
    cols = max(1, args.cols)
    if cols * card_w_px + (cols - 1) * 40 > mps.A4_PX[1]:
        sys.exit(f"卡片总宽 {cols * card_w_px / args.dpi * 2.54:.1f}cm 超出 A4 宽 21cm，"
                 "请减小 --card-w-cm 或 --cols")

    # 确定字体
    fonts = resolve_fonts(args.fonts, want=10)
    if not fonts:
        sys.exit("没有可用字体！请用 --fonts 指定字体文件路径。")
    if args.fonts and len(fonts) < 1:
        sys.exit("--fonts 指定的字体全部无效。")
    print(f"使用 {len(fonts)} 种字体: ")
    for i, (_, fam, sty) in enumerate(fonts):
        print(f"  [{i + 1:2d}] {fam} {sty}")

    # 加粗量
    if str(args.thicken).lower() == "auto":
        thicken = int(round(0.045 * digit_px))
    else:
        thicken = int(args.thicken)
    print(f"字符 {args.digit_cm}cm -> {digit_px}px, 合成加粗每侧 {thicken}px"
          f"({thicken / args.dpi * 2.54 * 10:.1f}mm, 0=不加粗)")

    # 渲染 10 数字 x N 字体，编号 start_seq + 字体序号
    cells = []                                    # (数字, 序号, 字体名, 数字图)
    for d in range(10):
        for i, (fp, fam, sty) in enumerate(fonts):
            seq = args.start_seq + i
            img, ink = render_digit_text(d, fp, digit_px, thicken=thicken)
            if img is None:
                print(f"  警告: 字体 {fam} {sty} 无法渲染数字 {d}，跳过")
                continue
            cells.append((d, seq, f"{fam} {sty}".strip(), img))
    if not cells:
        sys.exit("没有任何数字渲染成功，请检查字体。")

    rows_per_sheet = (3508 - 2 * 100) // (card_h_px + 40)   # 与 make_print_sheet 一致
    per_sheet = cols * rows_per_sheet
    n_sheets = (len(cells) + per_sheet - 1) // per_sheet
    seq_min, seq_max = args.start_seq, args.start_seq + len(fonts) - 1
    print(f"共 {len(cells)} 张卡片(每数字 {len(fonts)} 张), 每张A4放 {per_sheet} 张"
          f"({cols}列x{rows_per_sheet}行), 输出 {n_sheets} 张A4正面 + {n_sheets} 张背面, "
          f"编号 {0}-{seq_min:02d}..{9}-{seq_max:02d}")

    os.makedirs(args.out_dir, exist_ok=True)
    fronts, backs = [], []
    for s in range(n_sheets):
        part = cells[s * per_sheet:(s + 1) * per_sheet]
        front = compose_sheet_fonts(part, card_w_px, card_h_px, digit_px,
                                    cols=cols, frame=not args.no_frame)
        back = mps.compose_back_sheet([(lab, seq, None) for lab, seq, _, _ in part],
                                      card_w_px, card_h_px, cols=cols,
                                      flip=args.back_flip, text_scale=args.back_text_scale)
        fronts.append(front)
        backs.append(back)
        png_f = os.path.join(args.out_dir, f"sheet-{s + 1}-front.png")
        png_b = os.path.join(args.out_dir, f"sheet-{s + 1}-back.png")
        mps.save_gray_png(png_f, front)
        mps.save_gray_png(png_b, back)
        txt = os.path.join(args.out_dir, f"sheet-{s + 1}.txt")
        with open(txt, "w") as f:
            f.write(f"卡片 {args.card_w_cm}cm x {args.card_h_cm}cm"
                    f"(字符{args.digit_cm}cm, 加粗{thicken}px), 每行 {cols} 个, "
                    f"{args.dpi}dpi, 双面-{args.back_flip}, 字体版\n")
            for i, (lab, seq, fname, _) in enumerate(part):
                r, c = divmod(i, cols)
                f.write(f"第{r + 1}行 第{c + 1}列: 数字 {lab} ({fname}), "
                        f"背面编号 {lab}-{seq:02d}\n")
        print(f"已生成 {png_f} / {png_b}  (图例 {txt})")

    pdf = os.path.join(args.out_dir, "print_sheets.pdf")
    pages = [p for pair in zip(fronts, backs) for p in pair]   # 正1背1正2背2...
    mps.write_pdf(pdf, pages)
    print(f"已生成 {pdf}（{len(pages)} 页: 正1背1正2背2...）")
    print("打印: 自动双面选“双面打印-长边翻转”直接打（默认 --back-flip none 已对位）；")
    print("      手动双面请按翻纸方向加 --back-flip long/short 重新生成。")


if __name__ == "__main__":
    main()
