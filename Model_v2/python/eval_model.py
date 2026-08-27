#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用已保存模型评估指定扫描数据集（检查模型在新采集数据上的识别效果）
================================================================
加载 train_digit.py 保存的 *.pt（含 state_dict + meta），对选中的
datasets/<digit>_scan/<digit>-NN.csv 做预处理后逐样本预测，输出：
  总体准确率 / 每类准确率 / 混淆矩阵 / 误判清单（文件名 + 预测类）。

典型用途：新增/更新了扫描数据后，先用旧模型跑一遍，看新数据的识别
效果如何（本文档第 1 步"测试当前模型识别新的数字"）。

用法:
    python3 tools/eval_model.py --digits 0-4 --seq 21-30
                                          # 评估 datasets/{0..4}_scan/{0..4}-{21..30}.csv
    python3 tools/eval_model.py --digits 0-9 --seq 01-20
                                          # 评估全部旧样本
    python3 tools/eval_model.py --files a.csv,b.csv
                                          # 显式指定文件列表（逗号分隔）
    python3 tools/eval_model.py --model outputs/model_best.pt --digits 0-4 --seq 21-30

依赖: numpy, torch；预处理复用 tools/preprocess.py。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import process_csv


def parse_seq(spec):
    """'21-30' -> [21..30]；也支持 '5' 或 '1,3,5'。"""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def parse_digits(spec):
    """'0-4' -> ['0','1','2','3','4']；'0,2' -> ['0','2']。"""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return [str(i) for i in range(int(a), int(b) + 1)]
    return [d.strip() for d in spec.split(",") if d.strip()]


def collect_files(args):
    """返回 [(label, path), ...]（label 为字符串数字）。"""
    if args.files:
        out = []
        for p in args.files.split(","):
            p = p.strip()
            base = os.path.basename(p)
            digit = base.split("-")[0]
            out.append((digit, p))
        return out
    seqs = parse_seq(args.seq)
    out = []
    for d in parse_digits(args.digits):
        for n in seqs:
            f = os.path.join(args.src, f"{d}_scan", f"{d}-{n:02d}.csv")
            if os.path.isfile(f):
                out.append((d, f))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="用已保存模型评估扫描数据集")
    ap.add_argument("--model", default="outputs/model_best.pt",
                    help="模型文件 *.pt（默认 outputs/model_best.pt）")
    ap.add_argument("--src", default="datasets", help="数据集根目录（默认 datasets/）")
    ap.add_argument("--digits", default="0-4", help="数字范围，如 0-4 / 0,2,5（默认 0-4）")
    ap.add_argument("--seq", default="21-30",
                    help="文件名序号，如 21-30（默认 21-30，即新采集样本）")
    ap.add_argument("--files", default=None,
                    help="显式文件列表（逗号分隔，覆盖 --digits/--seq）")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    label_order = list(meta["label_order"])
    in_h = int(meta["in_h"])
    # 复用 train_digit.TinyCNN（同目录模块）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_digit import TinyCNN
    model = TinyCNN(in_h=in_h, in_w=int(meta["in_w"]),
                    n_classes=len(label_order))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    items = collect_files(args)
    if not items:
        sys.exit("没有匹配到任何 CSV 文件（检查 --digits/--seq/--src/--files）")

    idx_of = {d: i for i, d in enumerate(label_order)}
    preds, trues, names = [], [], []
    for digit, path in items:
        try:
            x = process_csv(path, in_h)
        except Exception as e:
            print(f"[跳过] {path}: {e}", file=sys.stderr)
            continue
        xb = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,8)
        logits = model(xb)
        p = int(logits.argmax(1).item())
        preds.append(p)
        trues.append(idx_of[digit])
        names.append(path)

    preds, trues = np.array(preds), np.array(trues)
    acc = float((preds == trues).mean())
    n = len(preds)

    print("=" * 64)
    print(f"模型: {args.model}")
    print(f"  meta: H={in_h}×{int(meta['in_w'])}, 类别={label_order}, "
          f"seed={meta.get('seed')}, mode={meta.get('mode', '-')}")
    print(f"样本: {n} 个 ({args.files or f'digits {args.digits}, seq {args.seq}'})")
    print(f"总体准确率: {acc:.1%} ({int((preds == trues).sum())}/{n})")
    print("-" * 64)
    print("每类准确率:")
    for i, d in enumerate(label_order):
        mask = trues == i
        if mask.sum() == 0:
            continue
        ca = float((preds[mask] == i).mean())
        print(f"  数字 {d}: {ca:.1%} ({int((preds[mask] == i).sum())}/{int(mask.sum())})")
    print("-" * 64)
    print("混淆矩阵（行=真实, 列=预测）:")
    cm = np.zeros((len(label_order), len(label_order)), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    hdr = "       " + " ".join(f"{d:>4}" for d in label_order)
    print(hdr)
    for i, d in enumerate(label_order):
        print(f"真实 {d}:  " + " ".join(f"{v:>4}" for v in cm[i]))
    print("-" * 64)
    bad = [(names[i], label_order[int(preds[i])]) for i in range(n)
           if preds[i] != trues[i]]
    if bad:
        print(f"误判 {len(bad)} 个:")
        for path, p in bad:
            print(f"  {path}  -> 预测 {p}")
    else:
        print("全部识别正确 ✓")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
