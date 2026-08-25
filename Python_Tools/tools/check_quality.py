#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 扫描数据质量体检
=========================
对 outputs/（或任意目录）里的每段 CSV 自动检查：
  1. 行数         —— 太短 = 扫太快 / 段被切，Y 分辨率不足
  2. 对比度 p95-p5 —— 太小 = 白纸与墨分不开（距离不稳时白纸读数也会飘高）
  3. 是否存在白纸低值 —— 整段找不到 < paper-thresh 的值 = 没扫到纸面(圈洞/边距)，
                       这类段对 0/6/8/9 这种带圈的字符几乎不可用
  4. 首/末行是否全高 —— 信息项（打印稿数字铺满卡片时首尾无白边属正常）
  5. 中段左右/中心均值 —— 信息项：0/6/8/9 应呈现"左右墨、中心纸"的圈结构

判定：任一硬性检查不通过 → [不可用]+原因；全部通过 → [可用]。

用法:
    python3 tools/check_quality.py                     # 检查 outputs/
    python3 tools/check_quality.py --src data/0        # 检查其它目录
    python3 tools/check_quality.py --min-rows 10 --min-contrast 1500
    python3 tools/check_quality.py --expect-ring       # 0/6/8/9 要求中段有纸面(圈洞)
    python3 tools/check_quality.py --paper-thresh 2500 # 白纸判定阈值(默认2000)

依赖: numpy（conda: conda activate my_project_ZSA）
"""
import argparse
import glob
import os
import sys

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="TCRT5000 扫描 CSV 质量体检")
    ap.add_argument("--src", default="outputs", help="CSV 目录（默认 outputs/）")
    ap.add_argument("--min-rows", type=int, default=10,
                    help="最少行数，低于视为扫太快/被切（默认 10）")
    ap.add_argument("--min-contrast", type=float, default=1500,
                    help="最小对比度 p95-p5（默认 1500）")
    ap.add_argument("--paper-thresh", type=float, default=2000,
                    help="白纸判定：值低于此视为真正纸面（默认 2000）")
    ap.add_argument("--ink-thresh", type=float, default=3500,
                    help="墨迹判定：值高于此视为墨（默认 3500）")
    ap.add_argument("--expect-ring", action="store_true",
                    help="对 0/6/8/9 等带圈字符：要求中段中心列存在纸面低值")
    return ap.parse_args()


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1:9].astype(float)


def check_file(path, args):
    """返回 (ok, reasons, info_dict)。"""
    vals = load_csv(path)
    n = vals.shape[0]
    reasons = []
    info = {"rows": n}

    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    contrast = p95 - p5
    info.update(p5=p5, p50=p50, p95=p95, contrast=contrast)

    if n < args.min_rows:
        reasons.append(f"行数太少({n}<{args.min_rows})")
    if contrast < args.min_contrast:
        reasons.append(f"对比度不足({contrast:.0f}<{args.min_contrast})")

    paper_min = vals.min()
    has_paper = paper_min < args.paper_thresh
    info["paper_min"] = paper_min
    if not has_paper:
        reasons.append(f"全程无白纸低值(min={paper_min:.0f}≥{args.paper_thresh:.0f})")

    # 中段圈结构（信息项）
    i1, i2 = int(n * 0.25), int(n * 0.75)
    mid = vals[i1:max(i1 + 1, i2)]
    m_l = mid[:, :2].mean()
    m_c = mid[:, 2:6].mean()
    m_r = mid[:, 6:8].mean()
    info.update(mid_left=m_l, mid_center=m_c, mid_right=m_r)
    if args.expect_ring and mid[:, 2:6].min() >= args.paper_thresh:
        reasons.append("中段中心列无纸面(圈洞缺失)")

    first_all_high = bool((vals[0] >= 4095 - 100).all())
    last_all_high = bool((vals[-1] >= 4095 - 100).all())
    info.update(first_all_high=first_all_high, last_all_high=last_all_high)

    ok = not reasons
    return ok, reasons, info


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.csv")))
    if not files:
        sys.exit(f"目录 {args.src} 里没有 CSV 文件")

    print(f"{'文件':<36}{'行数':>5}{'p5-p95':>11}{'纸面min':>9}"
          f"{'中段L/C/R':>16}  判定")
    n_ok = n_bad = 0
    for f in files:
        name = os.path.basename(f)
        ok, reasons, info = check_file(f, args)
        verdict = "[可用]" if ok else "[不可用]"
        if ok:
            n_ok += 1
        else:
            n_bad += 1
        lc = f"{info['mid_left']:.0f}/{info['mid_center']:.0f}/{info['mid_right']:.0f}"
        print(f"{name:<36}{info['rows']:>5}{info['p5']:.0f}-{info['p95']:.0f}"
              f"{info['paper_min']:>9.0f}{lc:>16} {verdict} {', '.join(reasons)}")

    print(f"\n合计: 可用 {n_ok} / 共 {len(files)}"
          f"{'  ← 可进入下一步重命名+划分' if n_ok else ''}")
    if n_bad:
        print(f"不可用 {n_bad} 段：建议按原因重新采集（固定距离垫片、放慢扫描、"
              f"确保扫到白纸/圈洞），或人工剔除后重跑本检查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
