#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 扫描样本质量体检 —— 找出建议重新扫描的样本
==================================================
两层检查：

1) 数据级检查（快，默认执行）：对 datasets/<digit>_scan/*.csv 逐样本统计
   行数、黑边裁剪行数、最大墨量、墨行占比、对比度，对照阈值打红旗：
   - 行数过少     rows_after < 25      （扫太快/被截断，Y 分辨率不足）
   - 墨量过弱     max_ADC < 3500       （笔画太淡，形状难辨）
   - 墨行占比过低 ink_ratio < 0.2      （整段几乎没墨）
   - 黑边过多     trimmed >= 8 行      （纸边进出不干净）
   - 对比度低     p95-p5 < 2400        （墨与纸分不开）

2) 模型级留出验证（--oof，慢：5 个 seed 各训练一次，约 7 分钟）：
   对每个样本统计它在"验证+测试"留出集合中被预测的次数、误判次数、
   平均真类置信度；被多次误判或置信度低的样本说明其形状与同类差异大
   （或与其他数字相近），最值得重扫。

用法:
    python3 tools/check_samples.py                # 仅数据级体检
    python3 tools/check_samples.py --oof          # 数据级 + 模型留出验证
    python3 tools/check_samples.py --min-flags 2  # 只列红旗 >= 2 的样本
输出: 控制台排名表 + outputs/sample_quality.json

依赖: numpy, torch(仅 --oof), matplotlib 不需要
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import scan_sources, trim_black_border

# ---- 数据级阈值（依据 200 个样本的分布实测标定）----
MIN_ROWS = 25          # 裁剪后行数下限
MIN_MAX_ADC = 3500     # 最大墨量下限（中位数 3657，弱墨 ~15% 样本低于此）
MIN_INK_RATIO = 0.2    # 墨行占比下限（中位数 0.70）
MAX_TRIM = 8           # 黑边裁剪行数上限（p90=6）
MIN_CONTRAST = 2400    # 对比度 p95-p5 下限（实测最低 2324）

OOF_SEEDS = [42, 1042, 2042, 3042, 4042]


# ================================================================ 数据级
def quality_metrics(vals):
    """返回 (metrics, flags)。vals 为裁剪前的 (N,8)。"""
    w = trim_black_border(vals)
    n_after = w.shape[0]
    trimmed = vals.shape[0] - n_after
    max_adc = w.max() if n_after else 0.0
    ink_rows = (w >= 3200).any(axis=1).sum() if n_after else 0
    ink_ratio = ink_rows / n_after if n_after else 0.0
    p5, p95 = np.percentile(w, [5, 95]) if n_after else (0, 0)
    contrast = p95 - p5

    flags = []
    if n_after < MIN_ROWS:
        flags.append(f"行数过少({n_after}<{MIN_ROWS})")
    if max_adc < MIN_MAX_ADC:
        flags.append(f"墨量过弱(max={max_adc:.0f}<{MIN_MAX_ADC})")
    if ink_ratio < MIN_INK_RATIO:
        flags.append(f"墨行占比过低({ink_ratio:.2f}<{MIN_INK_RATIO})")
    if trimmed >= MAX_TRIM:
        flags.append(f"黑边过多({trimmed}≥{MAX_TRIM}行)")
    if contrast < MIN_CONTRAST:
        flags.append(f"对比度低({contrast:.0f}<{MIN_CONTRAST})")

    m = {"rows_after": int(n_after), "rows_raw": int(vals.shape[0]),
         "trimmed": int(trimmed), "max_adc": float(max_adc),
         "ink_ratio": float(ink_ratio), "contrast": float(contrast)}
    return m, flags


# ================================================================ 模型级
def oof_analysis(items_by_digit, label_order, height=32, epochs=300,
                 patience=40, batch=16, lr=1e-3, verbose=False):
    """5 个 seed 留出验证：返回 {path: 聚合结果}。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from train_digit import ScanDataset, TinyCNN, make_split, VAL_N, TEST_N

    per_sample = defaultdict(list)      # path -> [{'y':..,'p':..,'correct':bool}]
    for seed in OOF_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        plan = make_split(items_by_digit, VAL_N, TEST_N, seed)

        train_samples, eval_samples = [], []
        for digit, files in items_by_digit.items():
            idx = label_order.index(digit)
            for f in files:
                (train_samples if plan[f] == "train" else eval_samples).append((f, idx))

        train_ds = ScanDataset(train_samples, height, augment=True)
        val_ds = ScanDataset(eval_samples, height, augment=False)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch)

        model = TinyCNN(in_h=height, in_w=8, n_classes=len(label_order))
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_acc, best_state, patience_left = -1.0, None, patience
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                n_c, n_t = 0, 0
                for xb, yb in val_loader:
                    n_c += (model(xb).argmax(1) == yb).sum().item()
                    n_t += xb.size(0)
                va = n_c / max(1, n_t)
            if va > best_acc:
                best_acc, best_state, patience_left = va, \
                    {k: v.clone() for k, v in model.state_dict().items()}, patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for path, y in eval_samples:
                ds = ScanDataset([(path, y)], height, augment=False)
                xb, yb = ds[0]
                prob = torch.softmax(model(xb.unsqueeze(0)), 1)[0]
                p = prob[y].item()
                pred = prob.argmax(0).item()
                per_sample[path].append({
                    "y": y, "p": p, "correct": bool(pred == y),
                    "pred": pred, "seed": seed,
                })
        if verbose:
            print(f"  [seed {seed}] 完成，val_best={best_acc:.3f}")

    # 聚合
    agg = {}
    for path, lst in per_sample.items():
        n = len(lst)
        n_wrong = sum(1 for r in lst if not r["correct"])
        agg[path] = {
            "seen": n, "wrong": n_wrong,
            "wrong_rate": n_wrong / n,
            "mean_true_prob": float(np.mean([r["p"] for r in lst])),
            "min_true_prob": float(np.min([r["p"] for r in lst])),
            "preds": [label_order[r["pred"]] for r in lst],
            "true": label_order[lst[0]["y"]],
        }
    return agg


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description="扫描样本质量体检，输出建议重扫清单")
    ap.add_argument("--src", default="datasets")
    ap.add_argument("--oof", action="store_true",
                    help="额外运行模型留出验证（5 个 seed，约 7 分钟）")
    ap.add_argument("--min-flags", type=int, default=0,
                    help="只列出红旗数 >= 该值的样本（默认全部）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = scan_sources(args.src)
    items_by_digit = defaultdict(list)
    for digit, f in items:
        items_by_digit[digit].append(f)
    label_order = sorted(items_by_digit)

    print(f"共 {len(items)} 个样本，逐样本体检中...")
    quality = {}
    for digit, f in items:
        vals = np.loadtxt(f, delimiter=",", skiprows=1)[:, 1:9].astype(float)
        m, flags = quality_metrics(vals)
        quality[f] = {"digit": digit, "flags": flags, **m}

    oof = {}
    if args.oof:
        print("模型留出验证（5 seed × 训练，约 7 分钟）...")
        oof = oof_analysis(items_by_digit, label_order, verbose=args.verbose)
        print("留出验证完成。")

    # ---- 汇总与排序 ----
    rows = []
    for digit, f in items:
        q = quality[f]
        o = oof.get(f, {})
        model_issue = (
            o.get("wrong_rate", 0) >= 0.5
            or (o.get("seen", 0) >= 2 and o.get("wrong", 0) >= 1
                and o.get("mean_true_prob", 1) < 0.35)
        ) if o else False
        rows.append({
            "file": f, "digit": q["digit"],
            "flags": q["flags"], "n_flags": len(q["flags"]),
            "model_issue": model_issue,
            "oof": o,
        })

    rows.sort(key=lambda r: (-r["n_flags"], -int(r["model_issue"]),
                             r["digit"], r["file"]))

    # ---- 打印排名表 ----
    print(f"\n{'文件':<30}{'行':>4}{'裁':>4}{'max':>6}{'墨比':>6}"
          f"{'留出误判':>9}{'真置信':>8}  红旗")
    for r in rows:
        q = quality[r["file"]]
        o = r["oof"]
        seen = o.get("seen", 0)
        oof_str = f"{o.get('wrong', 0)}/{seen}" if seen else "-"
        conf_str = f"{o.get('mean_true_prob', float('nan')):.2f}" if seen else "-"
        flag_str = ", ".join(r["flags"]) if r["flags"] else "—"
        star = " ★" if r["model_issue"] else ""
        print(f"{r['file']:<30}{q['rows_after']:>4}{q['trimmed']:>4}"
              f"{q['max_adc']:>6.0f}{q['ink_ratio']:>6.2f}"
              f"{oof_str:>9}{conf_str:>8}  {flag_str}{star}")

    # ---- 建议重扫清单 ----
    rec = [r for r in rows
           if r["n_flags"] >= max(1, args.min_flags) or r["model_issue"]]
    print(f"\n===== 建议重扫 {len(rec)}/{len(rows)} 个样本 =====")
    by_digit = defaultdict(list)
    for r in rec:
        by_digit[r["digit"]].append(r)
    for d in sorted(by_digit):
        print(f"\n数字 {d}（{len(by_digit[d])} 个）:")
        for r in by_digit[d]:
            o = r["oof"]
            extra = ""
            if o:
                extra = (f"  留出误判 {o['wrong']}/{o['seen']}"
                         f" 真类置信 {o['mean_true_prob']:.2f}"
                         f" 预测 {o['preds']}")
            print(f"  {r['file']}: {', '.join(r['flags']) if r['flags'] else '模型级存疑'}{extra}")

    # ---- 保存 JSON ----
    out = {"thresholds": {"MIN_ROWS": MIN_ROWS, "MIN_MAX_ADC": MIN_MAX_ADC,
                          "MIN_INK_RATIO": MIN_INK_RATIO, "MAX_TRIM": MAX_TRIM,
                          "MIN_CONTRAST": MIN_CONTRAST},
           "samples": {f: {"digit": q["digit"], "flags": q["flags"],
                           "metrics": {k: v for k, v in q.items()
                                       if k not in ("digit", "flags")},
                           "model": oof.get(f, {})}
                       for f, q in quality.items()},
           "recommended_rescan": [r["file"] for r in rec]}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_quality.json", "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: outputs/sample_quality.json")


if __name__ == "__main__":
    main()
