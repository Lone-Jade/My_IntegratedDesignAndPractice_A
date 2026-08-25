#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 STM32Cube.AI Studio 模型验证数据（.npz）
=============================================
STM32Cube.AI Studio 的"验证(Validation)"功能需要一份与模型输入严格对齐的
测试数据，格式为 NumPy .npz 归档，必须含键 x_test，可选键 y_test：

    x_test : float32, 形状 (N, 1, 32, 8)
             —— N 个样本，每样本 1 通道 × 32 行 × 8 传感器，
                与 model.onnx 的输入 1×1×32×8（batch=1）逐样本对齐；
                数值为预处理后的 [0,1] 灰度（墨=高值）。
    y_test : float32, 形状 (N, 1, 1, 10)
             —— one-hot 编码的真值（每类一列，共 10 类）。
                ST Edge AI `stedgeai validate` 要求 y_test 形状为
                (-1, 1, 1, num_classes)（可选项；--no-labels 可省略）。

数据来源：datasets/<digit>_scan/*.csv，经过与训练完全相同的预处理
（tools/preprocess.process_csv：无纸电平黑边裁剪 → 重采样 32 行 →
逐样本归一化 [0,1]）。默认取 14/3/3 划分的**测试集**（与训练一致、无泄漏）。

用法:
    python3 tools/export_ai_studio_data.py                                # 测试集 → outputs/ai_studio_test.npz
    python3 tools/export_ai_studio_data.py --split val                    # 验证集
    python3 tools/export_ai_studio_data.py --split all                    # 全部 200 个
    python3 tools/export_ai_studio_data.py --split all --samples 100      # 均衡采样 100 个
    python3 tools/export_ai_studio_data.py --out outputs/my_test.npz

依赖: numpy（无需 torch）
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import process_csv, scan_sources

DEFAULT_SPLIT_JSON = "outputs/split_seed42.json"


def load_split(split_json, split_name, src="datasets"):
    """从 split_seed<seed>.json 读划分，返回 [(digit, path), ...]（basename → 完整路径）。"""
    with open(split_json) as f:
        mani = json.load(f)               # {digit: {train:[fn], val:[fn], test:[fn]}}
    selected = []
    for digit, parts in sorted(mani.items()):
        for fn in parts.get(split_name, []):
            path = os.path.join(src, f"{digit}_scan", fn)
            if os.path.exists(path):
                selected.append((digit, path))
            else:
                print(f"[跳过] 找不到文件: {path}", file=sys.stderr)
    return selected


def main():
    ap = argparse.ArgumentParser(
        description="导出 STM32Cube.AI Studio 验证用 .npz（x_test/y_test）")
    ap.add_argument("--src", default="datasets",
                    help="数据集根目录（默认 datasets/）")
    ap.add_argument("--split", choices=["test", "val", "all"], default="test",
                    help="取哪个划分（默认 test：14/3/3 的测试集）")
    ap.add_argument("--split-json", default=None,
                    help="划分清单路径（默认 outputs/split_seed42.json，"
                         "--split all 时忽略）")
    ap.add_argument("--samples", type=int, default=0,
                    help="最多导出样本数（0=全部；>0 时按类均衡随机采样，seed 可复现）")
    ap.add_argument("--seed", type=int, default=42, help="采样随机种子（默认 42）")
    ap.add_argument("--out", default="outputs/ai_studio_test.npz",
                    help="输出 .npz 路径（默认 outputs/ai_studio_test.npz）")
    ap.add_argument("--no-labels", action="store_true",
                    help="不写 y_test（仅 x_test）")
    args = ap.parse_args()

    items_by_digit = defaultdict(list)
    for digit, path in scan_sources(args.src):
        items_by_digit[digit].append(path)

    # ---- 选取样本 ----
    if args.split == "all":
        selected = [(d, p) for d, ps in items_by_digit.items() for p in sorted(ps)]
    else:
        split_json = args.split_json or DEFAULT_SPLIT_JSON
        if not os.path.exists(split_json):
            sys.exit(f"找不到划分清单 {split_json}：请先运行 "
                     f"`tools/train_digit.py --ensemble 5` 生成，"
                     f"或改用 --split all")
        selected = load_split(split_json, args.split, args.src)
        if not selected:
            sys.exit(f"划分清单 {split_json} 中没有 {args.split} 集")

    # ---- 均衡采样限制数量 ----
    if args.samples and len(selected) > args.samples:
        rng = random.Random(args.seed)
        by_digit = defaultdict(list)
        for d, p in selected:
            by_digit[d].append((d, p))
        per = max(1, args.samples // len(by_digit))
        picked = []
        for d in sorted(by_digit):
            picked.extend(rng.sample(by_digit[d], min(per, len(by_digit[d]))))
        rest = [s for s in selected if s not in picked]
        picked.extend(rng.sample(rest, min(args.samples - len(picked), len(rest))))
        selected = sorted(picked)
        print(f"按 seed={args.seed} 均衡采样至 {len(selected)} 个样本")

    # ---- 预处理与打包 ----
    X, Y, names = [], [], []
    for digit, path in selected:
        img = process_csv(path)            # (32,8) float32, [0,1]
        X.append(img[None, ...])           # (1,32,8)
        Y.append(int(digit))
        names.append(f"{digit}_scan/{os.path.basename(path)}")

    x_test = np.stack(X).astype(np.float32)     # (N, 1, 32, 8)
    save_dict = {"x_test": x_test}
    if not args.no_labels:
        # ST Edge AI validate 期望 y_test 为 one-hot，形状 (N, 1, 1, num_classes)
        y_test = np.zeros((len(Y), 1, 1, len(items_by_digit)), dtype=np.float32)
        for i, lab in enumerate(Y):
            y_test[i, 0, 0, lab] = 1.0
        save_dict["y_test"] = y_test
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **save_dict)

    # ---- 打印摘要 ----
    print(f"已导出: {args.out}")
    print(f"  x_test: shape={x_test.shape} dtype={x_test.dtype} "
          f"值域=[{x_test.min():.3f}, {x_test.max():.3f}]")
    if not args.no_labels:
        counts = np.bincount(Y, minlength=len(items_by_digit))
        print(f"  y_test: shape={y_test.shape} dtype={y_test.dtype} (one-hot) "
              f"每类数量={dict(zip(sorted(items_by_digit), counts.tolist()))}")
    per_class = defaultdict(int)
    for d, _ in selected:
        per_class[d] += 1
    print(f"  每类样本数: {dict(sorted(per_class.items()))}")
    print("\n样本清单（前 10 个）:")
    for n in names[:10]:
        print(f"  {n}")
    if len(names) > 10:
        print(f"  ... 共 {len(names)} 个")
    print("\n在 STM32Cube.AI Studio 中：导入 model.onnx → 验证(Validation) "
          "→ 选择该 .npz 文件即可得到参考模型/量化模型的识别率对比。")


if __name__ == "__main__":
    main()
