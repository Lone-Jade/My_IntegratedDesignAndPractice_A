#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 扫描数据预处理管线
============================
把 scan_view.py 保存的 CSV（row, ch0..ch7）转换为可直接喂给 CNN 的
固定尺寸张量，处理人工扫描的三大问题：
  1. 黑边/楔形黑边伪影 —— 按"无纸电平（≈4095）"裁剪首/尾缘带，
     可去除不完整的黑边（同一行部分通道无纸、部分通道仍为纸面）；
  2. 行数不统一（25~88 行）—— Y 方向重采样到固定高度 H；
  3. 幅度/距离差异 —— 逐样本裁剪到 [p1,p99] 后 min-max 归一化到 [0,1]。

极性（与 show_scan.py 默认一致）：白纸 = 低值(~250)，黑墨 = 高值(~3400~4095)，
归一化后保持"墨 = 高值"。

用法:
    python3 tools/preprocess.py --preview                    # 每类 3 个样本拼网格图
    python3 tools/preprocess.py --preview --out-grid outputs/preprocess_grid.png
    python3 tools/preprocess.py --src datasets --height 32   # 只打印统计

本文件被 tools/train_digit.py 以模块方式 import，核心函数：
    load_csv(path)          -> np.ndarray (N,8) float
    trim_black_border(vals) -> np.ndarray
    resize_rows(vals, H)    -> np.ndarray (H,8)
    normalize(vals)         -> np.ndarray [0,1]
    process_csv(path, H)    -> np.ndarray (H,8) float32

依赖: numpy, matplotlib（仅 --preview 需要）
"""
import argparse
import glob
import os
import sys

import numpy as np

# ---- 可调阈值 ----
INK_THR = 3500          # 通道值 >= 此值视为墨（高值）
BORDER_FRAC = 6 / 8     # 一行中 >= 此比例的通道为墨 => 判定为黑边/伪影行
NO_PAPER_LEVEL = 4095   # ADC 满量程（无纸时读数）
NO_PAPER_MARGIN = 100   # 无纸判定裕量，与 scan_view.py 分段一致
N_CHANNELS = 8


# ---------------------------------------------------------------- 加载
def load_csv(path):
    """读取 CSV，返回 (N,8) float 矩阵（丢弃 row 列）。"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 1 + N_CHANNELS:
        raise ValueError(f"列数不足: 期望 row+8通道，实际 {data.shape[1]}")
    return data[:, 1:1 + N_CHANNELS].astype(float)


# ---------------------------------------------------------------- 黑边裁剪
def trim_black_border(vals, no_paper_margin=NO_PAPER_MARGIN):
    """裁剪首/尾"黑边/楔形黑边"缘带（v2，基于无纸电平）。

    背景：纸片进入/离开传感器视野时，若纸边与传感器直线不垂直（人工扫描
    常见），会形成**不完整黑边**——同一行里一部分通道无纸（≈4095）、
    一部分通道仍是纸面（低值）。旧规则（一行中 ≥6/8 通道 ≥3500）只能裁掉
    接近全黑的行，这些"楔形"行会被保留，污染图像。

    新规则：一行中**任一通道 ≥ 4095 - no_paper_margin（无纸电平）**即视为
    含无纸的行；从首/尾连续裁剪这些行（严格连续，无间隙容错）。
    已在全部 100 个 CSV 上验证：裁剪后保留区**无纸行残留 = 0、误裁纯纸行 = 0**，
    即所有楔形黑边被完整清除，真实笔画（深墨行一般 3400~3900，极少达 4095）
    不受影响。

    返回裁剪后的矩阵；全部为缘带时返回空矩阵。
    """
    if vals.shape[0] == 0:
        return vals
    np_thr = NO_PAPER_LEVEL - no_paper_margin
    np_row = (vals >= np_thr).any(axis=1)
    n = vals.shape[0]

    start = 0
    while start < n and np_row[start]:
        start += 1
    end = n
    while end > start and np_row[end - 1]:
        end -= 1

    return vals[start:end]


# ---------------------------------------------------------------- 重采样
def resize_rows(vals, H):
    """Y 方向线性重采样到固定高度 H 行，X（8 通道）保持不变。

    模拟做法：把行号归一化到 [0,1]，再插值到 H 个等距位置。
    速度扫描快→行数少→重采样后仍保持形状比例；慢→行数多→取整平滑。
    """
    n = vals.shape[0]
    if n == H:
        return vals
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, H)
    out = np.empty((H, vals.shape[1]), dtype=vals.dtype)
    for c in range(vals.shape[1]):
        out[:, c] = np.interp(dst, src, vals[:, c])
    return out


# ---------------------------------------------------------------- 归一化
def normalize(vals, low_p=1.0, high_p=99.0):
    """逐样本归一化：裁剪 [p1, p99] 后 min-max 到 [0,1]。

    逐样本（不用全局统计）→ 训练/验证/测试各自独立缩放，
    不会把测试集信息泄漏进训练；同时吸收扫描距离造成的整体明暗差异。
    """
    lo, hi = np.percentile(vals, [low_p, high_p])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip(vals, lo, hi)
    return (out - lo) / (hi - lo)


# ---------------------------------------------------------------- 完整管线
def process_csv(path, H=32):
    """一条 CSV -> (H,8) float32，取值 [0,1]，墨=高值。"""
    vals = load_csv(path)
    vals = trim_black_border(vals)
    if vals.shape[0] < 2:
        raise ValueError(f"黑边裁剪后行数不足: {vals.shape[0]}")
    vals = resize_rows(vals, H)
    vals = normalize(vals)
    return vals.astype(np.float32)


# ---------------------------------------------------------------- 中文字体
def setup_cjk_font():
    """为 matplotlib 选择一款系统中文字体（同 show_scan.py）。"""
    from matplotlib import font_manager
    import matplotlib
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "Source Han Sans CN", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "AR PL UMing CN", "Droid Sans Fallback",
    ]
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        if path:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    print("[字体] 警告：未找到中文字体，中文标签可能显示为方块。", file=sys.stderr)
    return None


# ---------------------------------------------------------------- 统计/预览
def scan_sources(src, pattern="*_scan"):
    """返回 [(digit, path), ...]，按 (数字, 文件名) 排序。"""
    items = []
    for d in sorted(glob.glob(os.path.join(src, pattern))):
        if not os.path.isdir(d):
            continue
        digit = os.path.basename(d).split("_")[0]
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            items.append((digit, f))
    return items


def print_stats(src):
    items = scan_sources(src)
    print(f"找到 {len(items)} 个 CSV")
    rows_all, trim_all, bad = [], 0, 0
    for digit, f in items:
        try:
            vals = load_csv(f)
        except Exception as e:
            bad += 1
            print(f"[跳过] {f}: {e}", file=sys.stderr)
            continue
        rows_all.append(vals.shape[0])
        trimmed = vals.shape[0] - trim_black_border(vals).shape[0]
        trim_all += trimmed
    rows_all = np.array(rows_all)
    print(f"行数: min={rows_all.min()} max={rows_all.max()} "
          f"mean={rows_all.mean():.1f} median={np.median(rows_all):.0f}")
    print(f"黑边裁剪总行数: {trim_all}")


def make_preview(src, H, out_path):
    """每类取前 3 个样本，处理成 (H,8) 后拼成网格灰度图。"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("缺少 matplotlib：pip install matplotlib")

    items = scan_sources(src)
    digits = sorted({d for d, _ in items})
    per_digit = {}
    for d, f in items:
        per_digit.setdefault(d, []).append(f)
    setup_cjk_font()
    cols = 3
    n_digits = len(digits)
    fig, axes = plt.subplots(n_digits, cols, figsize=(cols * 1.6, n_digits * 1.6))
    axes = np.atleast_2d(axes)
    for i, d in enumerate(digits):
        for j in range(cols):
            ax = axes[i, j]
            f = per_digit[d][j]
            img = process_csv(f, H)
            # gray_r: 高值(墨)=黑、低值(白纸)=白，与实物极性一致
            ax.imshow(img, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
                      origin="upper")
            ax.set_title(f"digit {d}: {os.path.basename(f)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"预处理后样本 (H={H}, 8 通道, 墨=黑)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"已保存预处理预览图: {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="TCRT5000 扫描数据预处理")
    ap.add_argument("--src", default="datasets", help="数据集根目录（默认 datasets/）")
    ap.add_argument("--height", type=int, default=32, help="重采样高度 H（默认 32）")
    ap.add_argument("--preview", action="store_true", help="生成预处理网格预览图")
    ap.add_argument("--out-grid", default="outputs/preprocess_grid.png",
                    help="预览图输出路径")
    args = ap.parse_args()

    if args.preview:
        make_preview(args.src, args.height, args.out_grid)
    else:
        print_stats(args.src)


if __name__ == "__main__":
    main()
