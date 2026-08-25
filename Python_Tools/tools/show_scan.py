#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 扫描数据 颜色深度示意图
================================
读取 scan_view.py 保存的 CSV（row,ch0..ch7），渲染成 2D 深度图：
    X 轴 = 8 路传感器 (CH0..CH7)
    Y 轴 = 扫描行（纸片移动方向，首行在上）
    极性（默认与纸片实物一致）: 白纸(低值)=白/亮、黑墨(高值)=黑/暗；
    --invert 反相，显示传感器原始极性（白纸=暗、黑墨=亮）。

用法（单文件 / 批量）:
    python3 show_scan.py outputs/xxx.csv                     # 灰度深度图
    python3 show_scan.py 文件.csv --invert                   # 反相(原始极性)
    python3 show_scan.py 文件.csv --thresh 2000              # 纯黑白二值化（墨=黑、纸=白）
    python3 show_scan.py 文件.csv --cmap viridis             # 需要彩色时
    python3 show_scan.py 文件.csv --out 图.png --no-show     # 只存图不弹窗
    # 批量：多个文件或目录，每文件各存一张 PNG（默认与 CSV 同目录同名 .png）
    python3 show_scan.py outputs/a.csv outputs/b.csv
    python3 show_scan.py outputs/                            # 目录内全部 CSV
    python3 show_scan.py outputs/ --out-dir pics/            # 批量存到指定目录
    python3 show_scan.py outputs/ --thresh 2000              # 批量二值化

批量时不弹窗；--out 仅用于单个文件。

依赖: pip install numpy matplotlib
中文字体: 找不到时中文标签会显示为方块，可安装 fonts-noto-cjk
          (Ubuntu: sudo apt install fonts-noto-cjk；装后清 matplotlib 缓存
           rm -rf ~/.cache/matplotlib)
"""
import sys
import argparse
import glob
import os

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def setup_cjk_font():
    """为 matplotlib 选择一款系统中文字体；找不到则警告并保持默认。"""
    if not HAS_MPL:
        return
    from matplotlib import font_manager
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
            print(f"[字体] 使用中文字体: {name}")
            return
    print("[字体] 警告：未找到中文字体，中文标签可能显示为方块。"
          "建议安装 fonts-noto-cjk 后清除 matplotlib 缓存重试。", file=sys.stderr)


def load_csv(path):
    """读取 CSV：第一列 row 号，第 2~9 列 CH0..CH7。
    返回 (rows:int数组, vals:float矩阵[行数][8])"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError(
            f"CSV 格式不对: 期望 'row,ch0..ch7'，实际形状 {data.shape}")
    rows = data[:, 0].astype(int)
    vals = data[:, 1:9].astype(float)
    return rows, vals


def plot_depth(rows, vals, args, title_path):
    n = vals.shape[0]
    fig_h = max(4.0, n * 0.10 + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    if args.thresh is not None:
        # 二值化（默认与实物一致：纸=白、墨=黑；--invert 反相）
        b = (vals < args.thresh) if not args.invert else (vals >= args.thresh)
        im = ax.imshow(b.astype(float), aspect="auto", cmap="gray",
                       origin="upper", vmin=0, vmax=1)
        cb_label = f"二值 (阈值 {args.thresh:g})"
    else:
        # 灰度：默认用反色映射 gray_r（低值=白纸=白、高值=墨=黑），
        # 色标仍显示真实 ADC 值；--invert 用原色映射（传感器原始极性）
        cmap = args.cmap if args.invert else args.cmap + "_r"
        im = ax.imshow(vals, aspect="auto", cmap=cmap, origin="upper",
                       vmin=args.vmin, vmax=args.vmax)
        cb_label = "ADC 值 (0-4095)"

    ax.set_xlabel("传感器通道 (CH0..CH7)")
    ax.set_ylabel("扫描行（Y 方向 = 纸片移动）")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"CH{i}" for i in range(8)])
    step = max(1, n // 20)
    ticks = list(range(0, n, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(rows[i]) for i in ticks])

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cb_label)
    fig.suptitle(os.path.basename(title_path))
    fig.tight_layout()
    return fig


def expand_inputs(inputs):
    """把 文件/目录 混合参数展开为 CSV 文件列表（去重保序）。"""
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.csv")))
        elif p.endswith(".csv"):
            files.append(p)
    seen, out = set(), []
    for f in files:
        r = os.path.realpath(f)
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out


def render_one(path, args, interactive):
    """渲染单个 CSV 并保存 PNG。interactive=True 时才弹窗。"""
    rows, vals = load_csv(path)
    fig = plot_depth(rows, vals, args, path)
    if args.out:
        out = args.out
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir,
                           os.path.splitext(os.path.basename(path))[0] + ".png")
    else:
        out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=args.dpi)
    print(f"已保存深度图: {out}")
    print(f"  数据: {vals.shape[0]} 行 × 8 通道, 值域 [{vals.min():.0f}, {vals.max():.0f}]")
    if interactive and not args.no_show:
        plt.show()
    plt.close(fig)          # 批量时及时释放内存


def main():
    ap = argparse.ArgumentParser(description="TCRT5000 扫描数据颜色深度图（支持批量）")
    ap.add_argument("inputs", nargs="+",
                    help="CSV 文件或目录（可多个；目录=其中全部 CSV）")
    ap.add_argument("--cmap", default="gray",
                    help="颜色映射: gray/viridis/jet/... (默认 gray 灰度)")
    ap.add_argument("--invert", action="store_true",
                    help="反相：显示传感器原始极性（白纸=暗、黑墨=亮）")
    ap.add_argument("--thresh", type=float, default=None,
                    help="二值化阈值：默认 值>=阈值为墨(黑)、值<阈值为纸(白)；"
                         "与 --invert 组合时反相")
    ap.add_argument("--vmin", type=float, default=0, help="颜色范围下限 (默认 0)")
    ap.add_argument("--vmax", type=float, default=4095, help="颜色范围上限 (默认 4095)")
    ap.add_argument("--out", help="单文件时输出 PNG 路径（批量时不可用）")
    ap.add_argument("--out-dir", help="批量输出目录（默认与各 CSV 同目录）")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-show", action="store_true", help="只保存图片，不弹窗")
    args = ap.parse_args()

    if not HAS_MPL:
        sys.exit("缺少 matplotlib：请先安装  pip install numpy matplotlib")

    setup_cjk_font()
    files = expand_inputs(args.inputs)
    if not files:
        sys.exit(f"没有找到 CSV 文件: {args.inputs}")
    if args.out and len(files) > 1:
        sys.exit("--out 仅用于单个文件；批量请用 --out-dir 或去掉 --out")
    if args.thresh is not None:
        print(f"二值化阈值 {args.thresh:g}：黑=墨(值≥阈值)，白=纸(值<阈值)")

    interactive = (len(files) == 1)
    ok = 0
    for f in files:
        try:
            render_one(f, args, interactive)
            ok += 1
        except Exception as e:
            print(f"[跳过] {os.path.basename(f)}: {e}", file=sys.stderr)
    print(f"完成: {ok}/{len(files)} 张")
    if ok < len(files):
        sys.exit(1)


if __name__ == "__main__":
    main()
