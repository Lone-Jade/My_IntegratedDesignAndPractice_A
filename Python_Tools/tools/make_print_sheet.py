#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 打印稿生成器（正面字卡 + 背面编号，双面打印）
=====================================================
把 MNIST 手写数字制作成可打印、可裁剪的 A4 稿：
  * 正面：虚线框为 9x6cm（默认），字符按内容包围盒等比放大到 5x5cm、居中
           —— 左右各留 2cm、上下各留 0.5cm 空白；卡片逐张过扫描导轨，
           两侧空白保证导轨不压到笔画；每行 2 个卡片、网格居中，
           沿虚线框裁剪即得 9x6cm 卡片；
  * 背面：与正面卡片"对位"打印编号，格式 `数字-序号`（如 3-07），
           编号已按翻页方式预先镜像，双面打印后正好落在对应卡片背面；
  * 先生成 300dpi PNG，再打包成单个 PDF（页序：正面1、背面1、正面2、背面2...）。

流程: MNIST(白字黑底 28x28) -> 反色(黑字白底) -> 二值化 -> 膨胀加粗
      -> 内容框等比放大到 5x5cm -> 放入 9x6cm 卡片(画虚线框, 四周留白)
      -> A4 居中排版(每行2个) -> 生成背面编号页 -> 存 PNG -> 转 PDF
说明: 加粗=笔画外扩1~2px，保证打印后笔画宽度 >= 传感器间距(~0.65cm)。
     膨胀会把 0/6/8/9 的圈糊成实心圆，脚本默认在膨胀后把原始就存在的圈
     重新打穿（--no-keep-holes 可关闭），保证圈始终开口。
     字符渲染默认用"原始灰度图"作渲染源（--smooth original，现在的方法）：
     灰度膨胀加粗 + 面积平滑放大，边沿软、无小方块；
     也可用"二值化的图"作渲染源（--smooth binary，之前的方法）：
     二值化+膨胀+最近邻，边沿为小方块。

用法:
    python3 make_print_sheet.py --per-digit 10        # 每类10个(共100张卡片)
    python3 make_print_sheet.py --dilate 2            # 加粗2px（推荐1~2）
    python3 make_print_sheet.py --smooth original    # 原始灰度图渲染(默认, 现在的方法)
    python3 make_print_sheet.py --smooth binary      # 二值化图渲染(之前的方法)
    python3 make_print_sheet.py --digit-cm 5          # 字符5x5cm(默认)
    python3 make_print_sheet.py --card-w-cm 9 --card-h-cm 6
                                                      # 虚线框9x6cm(默认)
    python3 make_print_sheet.py --cols 2              # 每行2个卡片(默认)
    python3 make_print_sheet.py --no-frame            # 不打印卡片虚线裁切框
    python3 make_print_sheet.py --back-flip none      # 自动双面默认; 手动双面用 long/short
    python3 make_print_sheet.py --source torch        # 用 torchvision 读取(推荐，绕开 gz 镜像)
    python3 make_print_sheet.py --source gz           # 用 gz 镜像下载（默认）
    python3 make_print_sheet.py --data-dir ./mnist    # MNIST 数据目录
    python3 make_print_sheet.py --out-dir ./print     # 输出目录

双面打印:
    推荐直接用 print_sheets.pdf 双面打印（页序 正1背1正2背2...，驱动自动配对）。
    * 打印机支持自动双面（对话框选"双面打印-长边翻转"）:
      用默认 --back-flip none 即可，驱动会自动补偿背面朝向，
      编号正好落在对应卡片背面，无需任何镜像；
    * 手动双面（打完正面把纸翻过来再打背面，无驱动补偿）:
      翻纸绕长边 -> --back-flip long（水平镜像抵消）；
      翻纸绕短边 -> --back-flip short（垂直镜像抵消）。
    不确定时：先只打印 sheet-1-front / sheet-1-back 一张试印，
    若编号落到别的卡片（错列/错行），换 --back-flip long / none / short 重试。

输出:
    print/sheet-1-front.png, sheet-1-back.png ...   (300dpi A4)
    print/print_sheets.pdf                          (页序 正1背1正2背2...)
    print/sheet-1.txt ...                           (每张图例：行列位置 = 数字-序号)

依赖: pip install numpy   (torch 模式另需: pip install torch torchvision)
"""
import sys
import os
import gzip
import argparse
import struct
import zlib
import urllib.request

import numpy as np

MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
# 镜像按优先级排列（原 yann.lecun.com 地址已失效，放最后兜底）
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "http://yann.lecun.com/exdb/mnist",
]
A4_PX = (3508, 2480)          # 300dpi A4 纵向 (高, 宽)
A4_PT = (595.28, 841.89)      # A4 尺寸（pt），PDF MediaBox
UA = {"User-Agent": "Mozilla/5.0 (make_print_sheet)"}

# 内置 5x7 点阵字体（数字 + 短横线），用于背面编号，避免依赖字体库
FONT_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
}


# ---------- MNIST 读取 ----------
def _download(url, dest):
    """优先 urllib，失败则退回系统 curl/wget。返回是否成功。"""
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return os.path.getsize(dest) > 0
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
    import shutil, subprocess
    if shutil.which("curl"):
        return subprocess.call(["curl", "-sL", "--max-time", "300", "-o", dest, url]) == 0
    if shutil.which("wget"):
        return subprocess.call(["wget", "-q", "-O", dest, url]) == 0
    return False


def ensure_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for name in MNIST_FILES:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            paths[name] = p
            continue
        ok = False
        for base in MNIST_MIRRORS:
            url = f"{base}/{name}"
            try:
                print(f"下载 {name} <- {url}", flush=True)
                if _download(url, p):
                    ok = True
                    break
                print("  文件为空，尝试下一个镜像...", flush=True)
            except Exception as e:
                print(f"  失败({e.__class__.__name__}: {e})，尝试下一个镜像...", flush=True)
        if not ok:
            sys.exit(f"下载 {name} 失败（所有镜像均不可用）。\n"
                     "请手动下载后放入数据目录（如 ./mnist/）：\n"
                     "  https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz\n"
                     "  https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")
        paths[name] = p
    return paths


def parse_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"images magic 不对: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def parse_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        assert magic == 2049, f"labels magic 不对: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(data_dir):
    paths = ensure_mnist(data_dir)
    imgs = parse_idx_images(paths["train-images-idx3-ubyte.gz"])
    labs = parse_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    return imgs, labs


def load_mnist_torch(data_dir):
    """用 torchvision 下载/读取 MNIST（绕开 gz 镜像，网络更稳）。
    数据缓存到 <data_dir>/MNIST/raw/，再次运行自动跳过下载。
    返回 (images (60000,28,28) uint8, labels (60000,) uint8)。"""
    from torchvision import datasets
    ds = datasets.MNIST(root=data_dir, train=True, download=True)
    return ds.data.numpy(), ds.targets.numpy()


# ---------- 预处理 ----------
def dilate_binary(b, k=3):
    """膨胀：3x3 窗口取最大，笔画向外扩 k//2 像素。"""
    from numpy.lib.stride_tricks import sliding_window_view
    p = k // 2
    bp = np.pad(b, p, mode="constant", constant_values=False)
    w = sliding_window_view(bp, (k, k)).reshape(b.shape[0], b.shape[1], -1)
    return w.max(axis=-1)


def label_components(mask):
    """4-连通标记 mask 的 True 连通域，返回 (labels, 数量)。纯 numpy，无 scipy 依赖。"""
    h, w = mask.shape
    labels = np.zeros(mask.shape, np.int32)
    cur = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                cur += 1
                labels[y, x] = cur
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels, cur


def enclosed_holes(digit, min_area=3):
    """找数字掩码里的封闭背景洞（0/6/8/9 的圈内），返回洞掩码（True=洞）。

    背景中不触碰图像边界的连通域就是圈内；面积 < min_area 的视为噪声忽略。
    """
    bg = ~digit
    labels, n = label_components(bg)
    sizes = np.bincount(labels.ravel())
    holes = np.zeros_like(digit)
    for i in range(1, n + 1):
        if sizes[i] < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        if ys.min() > 0 and ys.max() < digit.shape[0] - 1 and \
           xs.min() > 0 and xs.max() < digit.shape[1] - 1:
            holes |= (labels == i)
    return holes


def preprocess(img, dilate=1, keep_holes=True, min_hole_area=3):
    """反色 + 二值化 + 膨胀加粗，返回 0/255 的二值图（黑字=0，白底=255）。

    问题：膨胀会让 0/6/8/9 的圈被糊成实心黑圆（洞 < 2*加粗量就闭合）。
    解决：膨胀前先记录原始掩码里就存在的圈（enclosed_holes），
    膨胀后再把这些洞原样打穿，保证任何原始存在的圈都保持开口。
    """
    inv = 255 - img.astype(np.int16)        # 反色：黑底白字 -> 白底黑字
    digit = inv <= 128                       # 二值化：True=数字(黑)
    holes = enclosed_holes(digit, min_hole_area) if keep_holes else None
    for _ in range(max(0, dilate)):          # 只膨胀数字，笔画变粗
        digit = dilate_binary(digit)
    if holes is not None:
        digit = digit & ~holes               # 把被糊死的圈重新打通
    return np.where(digit, 0, 255).astype(np.uint8)   # 数字=0(黑)，背景=255(白)


def resize_nearest(img, new_h, new_w):
    """最近邻缩放（保持硬边），支持非正方形。"""
    yi = (np.arange(new_h) * img.shape[0]) // new_h
    xi = (np.arange(new_w) * img.shape[1]) // new_w
    return img[np.ix_(yi, xi)]


def resize_area(img, new_h, new_w):
    """面积平均（盒式滤波）缩放：每个目标像素取源区域的平均值。

    对 0/255 二值图放大后，边沿像素变成 0~255 的灰度过渡（抗锯齿），
    不再像最近邻那样出现整块的"小方块"台阶。可分离实现，纯 numpy。
    """
    if img.shape == (new_h, new_w):
        return img.astype(np.float64)

    def _resize_1d(vals, new_len):
        """1D 精确盒式滤波：目标 i 覆盖源区间 [i*L/N, (i+1)*L/N)。"""
        L = len(vals)
        if L == new_len:
            return vals.astype(np.float64)
        i = np.arange(new_len, dtype=np.float64)
        a = i * L / new_len
        b = (i + 1) * L / new_len
        i0 = np.floor(a).astype(int)
        i1 = np.ceil(b).astype(int) - 1
        c = np.concatenate([[0.0], np.cumsum(vals, dtype=np.float64)])
        s = c[i1 + 1] - c[i0]
        s = s - np.where(a > i0, (a - i0) * vals[i0], 0.0)
        s = s - np.where(b < i1 + 1, (i1 + 1 - b) * vals[i1], 0.0)
        return s / (b - a)

    tmp = np.empty((new_h, img.shape[1]), dtype=np.float64)
    for j in range(img.shape[1]):                      # 先纵向
        tmp[:, j] = _resize_1d(img[:, j], new_h)
    out = np.empty((new_h, new_w), dtype=np.float64)
    for i in range(new_h):                             # 再横向
        out[i, :] = _resize_1d(tmp[i, :], new_w)
    return out


def gray_dilate(img, r):
    """灰度膨胀（墨迹为暗色）：每像素取 (2r+1)x(2r+1) 窗口最小值。

    在 28x28 灰度图上做，轮廓沿原灰度平滑外扩（源是灰度、无台阶），
    与二值膨胀不同，不会产生小方块边沿。
    """
    if r <= 0:
        return img.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * r + 1
    bp = np.pad(img, r, mode="edge")
    w = sliding_window_view(bp, (k, k)).reshape(img.shape[0], img.shape[1], -1)
    return w.min(axis=-1)


def _place_crop(big, bbox_h, bbox_w, target):
    """big: 已按比例放大的图（包围盒在中心，可含少量留边）。
    取中心 bbox 区域，放到 target×target 画布正中。返回 0/255 图。"""
    s = target / max(bbox_h, bbox_w)
    ih2, iw2 = max(1, int(round(bbox_h * s))), max(1, int(round(bbox_w * s)))
    off_y = (big.shape[0] - ih2) // 2
    off_x = (big.shape[1] - iw2) // 2
    ink = big[off_y:off_y + ih2, off_x:off_x + iw2]
    out = np.full((target, target), 255.0, np.float64)
    yo, xo = (target - ih2) // 2, (target - iw2) // 2
    out[yo:yo + ih2, xo:xo + iw2] = ink
    return out


def render_digit(img, target, dilate=2, smooth="original", keep_holes=True):
    """渲染放大后的数字（target×target，0=墨）。

    加粗后的完整图形（含外扩边）按包围盒等比放大到 target 见方并居中，
    可见墨迹正好 target 见方。

    smooth 选择渲染源：
      "original"  现在的方法：直接用原始 MNIST 灰度图（自带抗锯齿笔迹）
                  灰度膨胀加粗 + 面积平滑放大，边沿软、无小方块(默认)；
      "binary"    之前的方法：用二值化的图片作渲染源
                  反色+二值化+二值膨胀加粗 + 最近邻放大，边沿为小方块。
    """
    inv = 255.0 - img.astype(np.float64)          # 反色: 0=墨最深
    digit_bin = inv <= 128.0                       # 二值化(用于找圈)
    holes = enclosed_holes(digit_bin, 3) if keep_holes else None

    if smooth == "binary":
        d = digit_bin.copy()
        for _ in range(max(0, dilate)):            # 二值膨胀加粗
            d = dilate_binary(d)
        if holes is not None:
            d = d & ~holes                         # 打穿圈
        ys, xs = np.nonzero(d)
        if len(ys) == 0:
            return np.full((target, target), 255, np.uint8)
        bbox_h, bbox_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        crop = np.where(d[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 0.0, 255.0)
        s = target / max(bbox_h, bbox_w)
        big = resize_nearest(crop, max(1, int(round(bbox_h * s))),
                             max(1, int(round(bbox_w * s))))
        return _place_crop(big, bbox_h, bbox_w, target).astype(np.uint8)

    # 原始灰度图渲染源: 灰度膨胀(轮廓随灰度平滑外扩) + 面积平滑放大(软边沿)
    thick = gray_dilate(inv, dilate)               # 加粗
    if holes is not None:                          # 打穿原始就存在的圈
        thick = np.where(holes, 255.0, thick)
    tb = thick < 128.0
    ys, xs = np.nonzero(tb)
    if len(ys) == 0:
        return np.full((target, target), 255, np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bbox_h, bbox_w = y1 - y0, x1 - x0
    pad = 2                                        # 留出灰度过渡边
    cy0, cy1 = max(0, y0 - pad), min(28, y1 + pad)
    cx0, cx1 = max(0, x0 - pad), min(28, x1 + pad)
    crop = thick[cy0:cy1, cx0:cx1]
    s = target / max(bbox_h, bbox_w)
    cnh, cnw = max(1, int(round(crop.shape[0] * s))), max(1, int(round(crop.shape[1] * s)))
    big = resize_area(crop, cnh, cnw)              # 平滑放大(边沿带灰度)
    return _place_crop(big, bbox_h, bbox_w, target).astype(np.uint8)


# ---------- 排版 ----------
def draw_dashed_rect(canvas, x0, y0, w, h, inset=0, dash=26, gap=16, lw=4, value=0):
    """在 canvas 上画 (x0,y0,w,h) 矩形的虚线边框，内缩 inset（默认0=沿边界）。

    虚线框外缘即 (x0,y0,w,h) 本身，测量虚线框尺寸即为 w×h。value=0 为黑色。"""
    x_l, x_r = x0 + inset, x0 + w - 1 - inset
    y_t, y_b = y0 + inset, y0 + h - 1 - inset
    # 上下边（水平虚线）
    x = x_l
    while x < x_r:
        seg = min(dash, x_r - x + 1)
        canvas[y_t:y_t + lw, x:x + seg] = value
        canvas[y_b - lw + 1:y_b + 1, x:x + seg] = value
        x += seg + gap
    # 左右边（垂直虚线）
    y = y_t
    while y < y_b:
        seg = min(dash, y_b - y + 1)
        canvas[y:y + seg, x_l:x_l + lw] = value
        canvas[y:y + seg, x_r - lw + 1:x_r + 1] = value
        y += seg + gap


def _grid_margins(card_w_px, card_h_px, cols, rows, gap):
    """按卡片数量把网格在 A4 上水平+垂直居中，返回 (margin_x, margin_y)。

    每行只放 cols 个卡片（默认2），网格居中。"""
    H, W = A4_PX
    total_w = cols * card_w_px + (cols - 1) * gap
    total_h = rows * card_h_px + (rows - 1) * gap
    return max(0, (W - total_w) // 2), max(0, (H - total_h) // 2)


def compose_sheet(cells, card_w_px, card_h_px, digit_px, cols=2, gap=40,
                  smooth="original", dilate=2, frame=True):
    """cells: [(label, seq, img28_raw)] -> 正面 A4 画布 (H,W) uint8。

    每行 cols 个卡片（默认2）；卡片 card_w_px x card_h_px（默认 9x6cm）：
    字符 render_digit 到 digit_px 见方（5x5cm）居中，左右/上下留白自动
    （默认左右各 2cm、上下各 0.5cm）；frame=True 时沿卡片边界画虚线裁切框。
    smooth: "original"|"binary" 见 render_digit。
    """
    H, W = A4_PX                # (高, 宽)
    rows = (len(cells) + cols - 1) // cols
    mx, my = _grid_margins(card_w_px, card_h_px, cols, rows, gap)
    canvas = np.full((H, W), 255, np.uint8)
    off_x = (card_w_px - digit_px) // 2
    off_y = (card_h_px - digit_px) // 2
    for i, (_, _, img) in enumerate(cells):
        r, c = divmod(i, cols)
        x0 = mx + c * (card_w_px + gap)
        y0 = my + r * (card_h_px + gap)
        if x0 + card_w_px > W or y0 + card_h_px > H:
            break
        card = np.full((card_h_px, card_w_px), 255, np.uint8)
        card[off_y:off_y + digit_px, off_x:off_x + digit_px] = \
            render_digit(img, digit_px, dilate=dilate, smooth=smooth)
        if frame:
            draw_dashed_rect(card, 0, 0, card_w_px, card_h_px, inset=0)   # 框=卡片边界
        canvas[y0:y0 + card_h_px, x0:x0 + card_w_px] = card
    return canvas


def render_text(text, scale=14, spacing=2):
    """用内置 5x7 点阵字体渲染文字，返回 0/255 图（0=黑字），已放大 scale 倍。"""
    char_w = 5 + spacing
    W = char_w * len(text) - spacing
    small = np.full((7, W), 255, np.uint8)
    for ci, ch in enumerate(text):
        g = FONT_5X7.get(ch)
        if g is None:
            continue
        for y in range(7):
            for x in range(5):
                if g[y][x] == "1":
                    small[y, ci * char_w + x] = 0
    return resize_nearest(small, 7 * scale, W * scale)


def compose_back_sheet(cells, card_w_px, card_h_px, cols=2, gap=40,
                       flip="none", text_scale=14):
    """cells: [(label, seq, img28)] -> 背面 A4 画布。

    编号 f"{label}-{seq:02d}" 放在与正面相同的卡片位置（同坐标系、同网格居中）。
    自动双面打印时驱动会补偿翻纸朝向，直接打印即可对位（flip="none"）。
    手动双面打印（无驱动补偿）时，翻纸方向会在物理上把背面内容左右/上下
    颠倒一次，需要预先镜像抵消：
      翻纸绕长边 -> flip="long"（水平镜像）；翻纸绕短边 -> flip="short"（垂直镜像）。
    """
    H, W = A4_PX
    rows = (len(cells) + cols - 1) // cols
    mx, my = _grid_margins(card_w_px, card_h_px, cols, rows, gap)
    canvas = np.full((H, W), 255, np.uint8)
    for i, (lab, seq, _) in enumerate(cells):
        r, c = divmod(i, cols)
        x0 = mx + c * (card_w_px + gap)
        y0 = my + r * (card_h_px + gap)
        if x0 + card_w_px > W or y0 + card_h_px > H:
            break
        text = render_text(f"{lab}-{seq:02d}", scale=text_scale)
        th, tw = text.shape
        y = y0 + (card_h_px - th) // 2
        x = x0 + (card_w_px - tw) // 2
        canvas[y:y + th, x:x + tw] = text
    if flip == "long":
        return canvas[:, ::-1].copy()
    if flip == "short":
        return canvas[::-1, :].copy()
    return canvas


# ---------- 输出：PNG / PDF（纯标准库，不依赖 matplotlib/PIL） ----------
def save_gray_png(path, arr):
    """保存 8bit 灰度 PNG。"""
    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def write_pdf(path, pages):
    """把若干 (H,W) 灰度页写成单个 A4 PDF（300dpi）。页序即传入顺序。"""
    W_pt, H_pt = A4_PT
    n = len(pages)
    page_ids = [3 + 3 * i for i in range(n)]
    img_ids = [4 + 3 * i for i in range(n)]
    cont_ids = [5 + 3 * i for i in range(n)]
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids [%s] /Count %d >>"
            % (" ".join(f"{p} 0 R" for p in page_ids), n)).encode(),
    }
    for i, page in enumerate(pages):
        H, W = page.shape
        data = zlib.compress(page.astype(np.uint8).tobytes(), 9)
        objs[img_ids[i]] = (
            f"<< /Type /XObject /Subtype /Image /Width {W} /Height {H} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")
        content = f"q {W_pt:.2f} 0 0 {H_pt:.2f} 0 0 cm /Im{img_ids[i]} Do Q".encode()
        objs[cont_ids[i]] = (f"<< /Length {len(content)} >>\nstream\n".encode()
                             + content + b"\nendstream")
        objs[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W_pt:.2f} {H_pt:.2f}] "
            f"/Resources << /XObject << /Im{img_ids[i]} {img_ids[i]} 0 R >> >> "
            f"/Contents {cont_ids[i]} 0 R >>").encode()
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for k in range(1, max(objs) + 1):
        offsets[k] = len(out)
        out += f"{k} 0 obj\n".encode() + objs[k] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {max(objs) + 1}\n0000000000 65535 f \n".encode()
    for k in range(1, max(objs) + 1):
        out += f"{offsets[k]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {max(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    with open(path, "wb") as f:
        f.write(out)


def main():
    ap = argparse.ArgumentParser(description="MNIST 打印稿生成器（正面字卡+背面编号，双面打印）")
    ap.add_argument("--per-digit", type=int, default=10, help="每类数字抽几个样本(默认10)")
    ap.add_argument("--dilate", type=int, default=2, help="笔画加粗像素数 1~3 (默认2)")
    ap.add_argument("--no-keep-holes", action="store_true",
                    help="不保留 0/6/8/9 的圈（默认会把膨胀糊死的圈重新打穿）")
    ap.add_argument("--no-frame", action="store_true",
                    help="不打印卡片虚线裁切框（默认打印）")
    ap.add_argument("--digit-cm", type=float, default=5.0,
                    help="字符最大尺寸 cm(默认5, 字符 5x5cm, 上下自动留白)")
    ap.add_argument("--card-w-cm", type=float, default=9.0,
                    help="虚线框宽 cm(默认9=5cm字符+左右各2cm留白)")
    ap.add_argument("--card-h-cm", type=float, default=6.0,
                    help="虚线框高 cm(默认6=5cm字符+上下各0.5cm留白)")
    ap.add_argument("--cols", type=int, default=2, help="每行卡片个数(默认2)")
    ap.add_argument("--smooth", choices=["original", "binary"], default="original",
                    help="字符渲染源: original=用原始灰度图渲染, 软边沿无小方块(默认, "
                         "现在的方法); binary=用二值化的图渲染, 最近邻方块(之前的方法)")
    ap.add_argument("--back-flip", choices=["long", "short", "none"], default="none",
                    help="背面镜像方式: none=自动双面驱动代偿(默认), "
                         "long=手动双面翻长边, short=手动双面翻短边")
    ap.add_argument("--back-text-scale", type=int, default=14, help="背面编号字号(点阵像素倍数, 默认14)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="随机种子，保证可复现")
    ap.add_argument("--source", choices=["auto", "torch", "gz"], default="auto",
                    help="数据来源: auto=有torchvision用torch否则gz(默认), torch=torchvision, gz=官方镜像")
    ap.add_argument("--data-dir", default="./mnist", help="MNIST 数据目录(torch 缓存到 <dir>/MNIST/raw/)")
    ap.add_argument("--out-dir", default="./print", help="输出目录")
    args = ap.parse_args()

    use_torch = args.source == "torch"
    if args.source == "auto":
        try:
            import torchvision  # noqa: F401
            use_torch = True
        except ImportError:
            use_torch = False

    os.makedirs(args.out_dir, exist_ok=True)
    if use_torch:
        imgs, labs = load_mnist_torch(args.data_dir)
        print(f"MNIST 已通过 torchvision 加载: {imgs.shape[0]} 张 {imgs.shape[1]}x{imgs.shape[2]}")
    else:
        imgs, labs = load_mnist(args.data_dir)
        print(f"MNIST 已通过 gz 镜像加载: {imgs.shape[0]} 张 {imgs.shape[1]}x{imgs.shape[2]}")

    # 尺寸换算（300dpi: 1cm ≈ 118px）: 字符 5x5cm 居中放 9x6cm 卡片
    digit_px = int(round(args.digit_cm / 2.54 * args.dpi))
    card_w_px = int(round(args.card_w_cm / 2.54 * args.dpi))
    card_h_px = int(round(args.card_h_cm / 2.54 * args.dpi))
    if digit_px > card_w_px - 20 or digit_px > card_h_px - 20:
        sys.exit(f"字符 {args.digit_cm}cm 大于卡片 {args.card_w_cm}x{args.card_h_cm}cm，"
                 "请减小 --digit-cm 或增大 --card-w-cm/--card-h-cm")
    cols = max(1, args.cols)
    if cols * card_w_px + (cols - 1) * 40 > A4_PX[1]:
        sys.exit(f"卡片总宽 {cols * card_w_px / args.dpi * 2.54:.1f}cm 超出 A4 宽 21cm，"
                 "请减小 --card-w-cm 或 --cols")
    side_blank = (card_w_px - digit_px) / 2 / args.dpi * 2.54   # 左右留白 cm
    top_blank = (card_h_px - digit_px) / 2 / args.dpi * 2.54    # 上下留白 cm

    rng = np.random.default_rng(args.seed)
    cells = []                                         # (数字, 该数字内序号, 28x28原始灰度图)
    for d in range(10):
        idx = np.where(labs == d)[0]
        pick = rng.choice(idx, size=min(args.per_digit, len(idx)), replace=False)
        for seq, j in enumerate(pick, start=1):
            cells.append((d, seq, imgs[j]))

    rows_per_sheet = (3508 - 2 * 100) // (card_h_px + 40)   # 纵向最多能放几行(6cm高=4行)
    per_sheet = cols * rows_per_sheet
    n_sheets = (len(cells) + per_sheet - 1) // per_sheet
    print(f"共 {len(cells)} 张卡片, 卡片 {card_w_px / args.dpi * 2.54:.1f}x{card_h_px / args.dpi * 2.54:.1f}cm"
          f"(字符{args.digit_cm}cm, 左右各留{side_blank:.1f}cm, 上下各留{top_blank:.1f}cm), "
          f"每张A4放 {per_sheet} 张({cols}列x{rows_per_sheet}行), "
          f"输出 {n_sheets} 张A4正面 + {n_sheets} 张背面")

    fronts, backs = [], []
    for s in range(n_sheets):
        part = cells[s * per_sheet:(s + 1) * per_sheet]
        front = compose_sheet(part, card_w_px, card_h_px, digit_px, cols=cols,
                              smooth=args.smooth, dilate=args.dilate,
                              frame=not args.no_frame)
        back = compose_back_sheet(part, card_w_px, card_h_px, cols=cols,
                                  flip=args.back_flip, text_scale=args.back_text_scale)
        fronts.append(front)
        backs.append(back)
        png_f = os.path.join(args.out_dir, f"sheet-{s + 1}-front.png")
        png_b = os.path.join(args.out_dir, f"sheet-{s + 1}-back.png")
        save_gray_png(png_f, front)
        save_gray_png(png_b, back)
        txt = os.path.join(args.out_dir, f"sheet-{s + 1}.txt")
        with open(txt, "w") as f:
            f.write(f"卡片 {card_w_px / args.dpi * 2.54:.1f}cm x {card_h_px / args.dpi * 2.54:.1f}cm"
                    f"(字符{args.digit_cm}cm, 左右留白{side_blank:.1f}cm, 上下留白{top_blank:.1f}cm), "
                    f"每行 {cols} 个, 加粗 {args.dilate}px, 300dpi, 双面-{args.back_flip}\n")
            for i, (lab, seq, _) in enumerate(part):
                r, c = divmod(i, cols)
                f.write(f"第{r + 1}行 第{c + 1}列: 数字 {lab}, 背面编号 {lab}-{seq:02d}\n")
        print(f"已生成 {png_f} / {png_b}  (图例 {txt})")

    pdf = os.path.join(args.out_dir, "print_sheets.pdf")
    pages = [p for pair in zip(fronts, backs) for p in pair]   # 正1背1正2背2...
    write_pdf(pdf, pages)
    print(f"已生成 {pdf}（{len(pages)} 页: 正1背1正2背2...）")
    print("打印: 自动双面选“双面打印-长边翻转”直接打（默认--back-flip none 已对位）；")
    print("      手动双面请按翻纸方向加 --back-flip long/short 重新生成。")
    print("      建议先试印 sheet-1，确认背面编号落在卡片框内再整批打印。")


if __name__ == "__main__":
    main()
