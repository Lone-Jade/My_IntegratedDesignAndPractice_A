#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRT5000 扫描数据 10 类数字（0~9）识别训练与可行性评估
====================================================
流程：
  1. 读 datasets/<digit>_scan/<digit>-NN.csv（每文件=一张独立纸片扫描一次）；
  2. 划分（默认 fixed，按文件名序号固定划分，可复现）：
       训练: 1-14 + 21-26 / 验证: 15-17 + 27-28 / 测试: 18-20 + 29-30
     0~9 每类 30 个文件（01~20 MNIST 打印体 + 21~30 字体版，新旧字体
     按比例混入各集合）；也可用 --split random
     回到"每类随机 14/3/3"；
  3. 预处理（复用 tools/preprocess.py：黑边裁剪→重采样→逐样本归一化）；
  4. 训练集在线数据增强（Y 拉伸/横向平移/幅度抖动/噪声，模拟人工扫描不统一）；
  5. 微型 2D CNN 训练（早停、保存最优与最终权重、每 epoch 训练过程数据）；
  6. 测试集最终评估：总体/每类准确率 + 混淆矩阵；
  7. 按测试准确率给出"扫描方案是否可行"的判定。

产物（默认写入 outputs/）：
    model_best.pt          验证集最优权重（含模型配置，可加载推理）
    model_final.pt         训练结束时的最终权重
    training_log.csv       每 epoch 训练过程数据
    curves.png             loss/acc 曲线（由 training_log.csv 绘制）
    confusion_matrix.png   测试集混淆矩阵
    split_fixed.json       固定划分清单（--split random 时为 split_seed<seed>.json）
    train_summary.json     汇总：各集合准确率、每类准确率、混淆矩阵、判定

用法:
    python3 tools/train_digit.py                          # 默认 fixed 划分, seed=42, repeats=1
    python3 tools/train_digit.py --repeats 5              # 5 个不同模型 seed 求测试准确率 mean±std
    python3 tools/train_digit.py --ensemble 5             # 多 seed 投票 + 权重平均（含 ONNX 导出与部署内存分析）
    python3 tools/train_digit.py --split random           # 旧式随机 14/3/3 划分
    python3 tools/train_digit.py --label-smoothing 0.1 --weight-decay 1e-4
    python3 tools/train_digit.py --height 64 --epochs 400
    python3 tools/train_digit.py --src datasets --out-dir outputs

依赖: numpy, torch(CPU 即可), matplotlib（仅画图用，--no-plots 可跳过）
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (load_csv, normalize, resize_rows, scan_sources,
                        setup_cjk_font, trim_black_border)

VAL_N = 3      # 每类验证样本数
TEST_N = 3     # 每类测试样本数


# ================================================================ 模型
class TinyCNN(nn.Module):
    """微型 2D CNN：输入 (1, H, 8)，三层卷积 + 池化 + 两层全连接。

    参数量约 2 万，小到后续可考虑部署 STM32G0。
    """

    def __init__(self, in_h=32, in_w=8, n_classes=10, drop=0.3):
        super().__init__()
        self.in_h, self.in_w, self.n_classes = in_h, in_w, n_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        h, w = in_h // 4, in_w // 4          # 两次 2×2 池化后
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * h * w, 32), nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ================================================================ 数据集
class ScanDataset(Dataset):
    """每个样本 = 一个 CSV 扫描。augment=True 时在线随机增强。"""

    def __init__(self, samples, H=32, augment=False):
        self.samples = samples          # [(path, label_idx), ...]
        self.H = H
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        vals = load_csv(path)
        vals = trim_black_border(vals)
        if vals.shape[0] < 2:
            raise ValueError(f"黑边裁剪后行数不足: {path}")
        if self.augment:
            vals = self._geom_augment(vals)      # 原始尺度上做几何增强
        vals = resize_rows(vals, self.H)
        vals = normalize(vals)
        if self.augment:
            vals = self._amp_augment(vals)       # 归一化后做幅度/噪声增强
        x = torch.from_numpy(vals).float().unsqueeze(0)   # (1, H, 8)
        return x, y

    def _geom_augment(self, vals):
        """几何增强（原始 ADC 尺度，归一化前）：
        - Y 向拉伸 0.8~1.25 倍（扫描快慢）；
        - X 向 ±1 通道平移（纸片横向错位，边缘复制填充）。
        不做水平翻转（数字不对称）。
        """
        s = np.random.uniform(0.8, 1.25)
        n2 = max(2, int(round(vals.shape[0] * s)))
        vals = resize_rows(vals, n2)
        sh = np.random.randint(-1, 2)
        if sh != 0:
            vals = np.pad(vals, ((0, 0), (1, 1)), mode="edge")
            vals = vals[:, 1 + sh:1 + sh + 8]
        return vals

    def _amp_augment(self, vals):
        """幅度增强（归一化后 [0,1] 尺度）：增益 0.9~1.1 + 高斯噪声 σ≈0.02，
        模拟扫描距离/抖动造成的整体明暗变化。"""
        g = np.random.uniform(0.9, 1.1)
        vals = vals * g + np.random.normal(0, 0.02, size=vals.shape)
        return vals


# ================================================================ 划分
def make_split(items_by_digit, val_n, test_n, seed):
    """每类洗牌后：前 val_n 张验证、接下来 test_n 张测试、其余训练。
    返回 plan: path -> 'train'|'val'|'test'。"""
    rng = random.Random(seed)
    plan = {}
    for digit in sorted(items_by_digit):
        files = sorted(items_by_digit[digit])
        if len(files) < val_n + test_n + 1:
            raise SystemExit(f"数字 {digit}: 只有 {len(files)} 个文件，"
                             f"不够划分 训练≥1+验证{val_n}+测试{test_n}")
        rng.shuffle(files)
        for f in files[:val_n]:
            plan[f] = "val"
        for f in files[val_n:val_n + test_n]:
            plan[f] = "test"
        for f in files[val_n + test_n:]:
            plan[f] = "train"
    return plan


def make_fixed_split(items_by_digit):
    """按文件名序号固定划分（不随机、完全可复现，适合新旧字体混合的数据集）：

        训练: 序号 1-14 或 21-26
        验证: 序号 15-17 或 27-28
        测试: 序号 18-20 或 29-30

    文件名格式 <digit>-NN.csv（NN = 01..30）。旧样本（01~20，MNIST 打印体）
    与新样本（21~30，字体版打印稿）在各集合中**按比例混合**，
    避免随机划分把某一字体整体留出。返回 plan: path -> 'train'|'val'|'test'。
    """
    def seq_of(path):
        base = os.path.basename(path)            # 如 "3-27.csv"
        return int(base.rsplit("-", 1)[1].split(".")[0])

    plan = {}
    for digit in sorted(items_by_digit):
        for f in items_by_digit[digit]:
            n = seq_of(f)
            if n <= 14 or 21 <= n <= 26:
                sp = "train"
            elif 15 <= n <= 17 or 27 <= n <= 28:
                sp = "val"
            elif 18 <= n <= 20 or 29 <= n <= 30:
                sp = "test"
            else:
                raise SystemExit(f"文件序号 {n} 超出固定划分规则: {f}"
                                 f"（应为 1-14/15-17/18-20/21-26/27-28/29-30）")
            plan[f] = sp
    return plan


def make_plan(split_mode, items_by_digit, val_n, test_n, seed):
    """按 split_mode 生成划分 plan；固定划分下忽略 seed（划分不随机）。"""
    if split_mode == "fixed":
        return make_fixed_split(items_by_digit)
    return make_split(items_by_digit, val_n, test_n, seed)


def split_descr(split_mode, val_n, test_n):
    if split_mode == "fixed":
        return "固定划分 训练1-14+21-26 / 验证15-17+27-28 / 测试18-20+29-30"
    return f"{val_n}+{test_n} (每类 训练/验证/测试)"


def split_manifest(plan, items_by_digit):
    """转为 {digit: {'train':[文件名...], 'val':[...], 'test':[...]}} 便于保存。"""
    mani = {}
    for digit, files in sorted(items_by_digit.items()):
        mani[digit] = {sp: sorted(os.path.basename(f)
                                  for f in files if plan[f] == sp)
                       for sp in ("train", "val", "test")}
    return mani


# ================================================================ 评估
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct = 0, 0
    loss_sum, preds, targets = 0.0, [], []
    for xb, yb in loader:
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        p = logits.argmax(dim=1)
        preds.append(p.numpy())
        targets.append(yb.numpy())
        correct += (p == yb).sum().item()
        total += xb.size(0)
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    return loss_sum / max(1, total), correct / max(1, total), preds, targets


def confusion_matrix(preds, targets, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, targets):
        m[t, p] += 1
    return m


# ================================================================ 训练
def train_model(train_samples, val_samples, model_seed, args, label_order,
                height, verbose=False):
    """在给定训练/验证样本上训练一个 TinyCNN（Label Smoothing + AdamW + 早停）。

    返回 (best_state, info)；best_state 为验证集最优权重（可加载推理）。
    依赖调用方事先设置 torch/numpy/random 全局种子（模型种子）。
    """
    t0 = time.time()
    train_ds = ScanDataset(train_samples, height, augment=True)
    val_ds = ScanDataset(val_samples, height, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = TinyCNN(in_h=height, in_w=8, n_classes=len(label_order))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_acc, best_epoch, best_state = -1.0, -1, None
    patience_left = args.patience
    log = []                       # 每 epoch: [epoch, tr_loss, tr_acc, va_loss, va_acc]

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)
        log.append([epoch, tr_loss, tr_acc, va_loss, va_acc])

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if (verbose and epoch % 10 == 0) or patience_left <= 0:
            print(f"  [seed {model_seed}] epoch {epoch:3d} "
                  f"tr_loss {tr_loss:.4f} tr_acc {tr_acc:.3f} "
                  f"va_loss {va_loss:.4f} va_acc {va_acc:.3f}")
        if patience_left <= 0:
            break

    return best_state, {"best_val_acc": float(best_acc),
                        "best_epoch": best_epoch,
                        "epochs_done": len(log),
                        "elapsed_s": round(time.time() - t0, 1),
                        "log": log}


def run_experiment(seed, args, items_by_digit, label_order,
                   out_dir, save_artifacts):
    """一次完整实验：划分→训练→测试。save_artifacts=True 时保存详细产物。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    plan = make_plan(args.split, items_by_digit, VAL_N, TEST_N, seed)
    samples = {"train": [], "val": [], "test": []}
    for digit, files in items_by_digit.items():
        idx = label_order.index(digit)
        for f in files:
            samples[plan[f]].append((f, idx))

    best_state, info = train_model(samples["train"], samples["val"], seed,
                                   args, label_order, args.height,
                                   verbose=args.verbose)
    model = TinyCNN(in_h=args.height, in_w=8, n_classes=len(label_order))
    model.load_state_dict(best_state)
    test_ds = ScanDataset(samples["test"], args.height, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    test_loss, test_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion)
    cm = confusion_matrix(test_preds, test_targets, len(label_order))
    per_class = {}
    for i, d in enumerate(label_order):
        denom = cm[i].sum()
        per_class[d] = (float(cm[i, i] / denom) if denom else float("nan"))

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "seed": seed,
        "n_train": len(samples["train"]),
        "n_val": len(samples["val"]),
        "n_test": len(samples["test"]),
        "epochs_done": info["epochs_done"],
        "best_epoch": info["best_epoch"],
        "best_val_acc": info["best_val_acc"],
        "elapsed_s": info["elapsed_s"],
        "n_params": n_params,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "per_class_acc": per_class,
        "confusion": cm.tolist(),
    }

    if save_artifacts:
        # ---- 模型 ----
        meta = {"in_h": args.height, "in_w": 8, "n_classes": len(label_order),
                "label_order": label_order, "seed": seed,
                "test_acc": result["test_acc"]}
        torch.save({"state_dict": best_state, "meta": meta},
                   os.path.join(out_dir, "model_best.pt"))
        torch.save({"state_dict": model.state_dict(), "meta": meta},
                   os.path.join(out_dir, "model_final.pt"))
        # ---- 训练过程数据 ----
        with open(os.path.join(out_dir, "training_log.csv"), "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            for row in info["log"]:
                f.write(",".join(f"{v:.6f}" for v in row) + "\n")
        # ---- 划分清单 ----
        split_name = (f"split_fixed.json" if args.split == "fixed"
                      else f"split_seed{seed}.json")
        with open(os.path.join(out_dir, split_name), "w") as f:
            json.dump(split_manifest(plan, items_by_digit), f,
                      ensure_ascii=False, indent=2)
        # ---- 曲线与混淆矩阵 ----
        try:
            plot_curves(info["log"], os.path.join(out_dir, "curves.png"))
            plot_confusion(cm, label_order,
                           os.path.join(out_dir, "confusion_matrix.png"))
        except ImportError:
            print("[警告] 缺少 matplotlib，跳过曲线/混淆矩阵图片生成",
                  file=sys.stderr)
    return result


# ================================================================ 集成/部署
@torch.no_grad()
def predict_probs(model, loader):
    """返回测试集每个样本的 softmax 概率 (n, n_classes)。"""
    model.eval()
    return torch.cat([torch.softmax(model(xb), 1) for xb, _ in loader]).numpy()


def majority_vote(probs_list):
    """N 个模型的多数投票；平票时取平均置信度最高的类。
    probs_list: [(n, n_classes) ...] → 预测类别 (n,)"""
    probs = np.stack(probs_list)            # (N, n, C)
    votes = probs.argmax(axis=2)            # (N, n)
    n = votes.shape[1]
    n_classes = probs.shape[2]
    preds = np.empty(n, dtype=int)
    for i in range(n):
        counts = np.bincount(votes[:, i], minlength=n_classes)
        top = np.flatnonzero(counts == counts.max())
        if len(top) == 1:
            preds[i] = top[0]
        else:
            preds[i] = top[int(probs[:, i, top].mean(axis=0).argmax())]
    return preds


def memory_analysis(model, in_h, in_w):
    """估算 fp32/int8 权重字节数与激活峰值（KB），供 STM32 部署参考。

    G0B1RBT6：Cortex-M0+（无 FPU）、128KB Flash、144KB SRAM。
    Cube.AI 运行时另需约 20~40KB Flash（int8 更小），报告里注明。
    """
    n_params = sum(p.numel() for p in model.parameters())
    h, w = in_h, in_w
    acts = [in_h * in_w,                    # 输入
            h * w * 8,                      # conv1 输出
            h * w * 16,                     # conv2 输出
            (h // 2) * (w // 2) * 16,       # pool1 后
            (h // 2) * (w // 2) * 32,       # conv3 输出
            (h // 4) * (w // 4) * 32]       # pool2 后
    peak = max(acts)
    return {"n_params": n_params,
            "weights_fp32_kb": round(n_params * 4 / 1024, 1),
            "weights_int8_kb": round(n_params / 1024, 1),
            "peak_activation_fp32_kb": round(peak * 4 / 1024, 1),
            "peak_activation_int8_kb": round(peak / 1024, 1)}


def export_onnx(model, out_path, in_h, in_w):
    """导出 eval 模式模型为 ONNX（输入 1×1×H×8），供 STM32Cube.AI 导入。

    需要 pip 包 onnx（可选依赖，仅导出时用）：
        pip install onnx
    优先用新导出器（torch.export），缺 onnxscript 时回退传统导出器
    （dynamo=False，只需 onnx 包）。
    """
    model.eval()
    x = torch.zeros(1, 1, in_h, in_w)
    kwargs = dict(input_names=["input"], output_names=["output"],
                  opset_version=13)
    missing = None
    try:
        torch.onnx.export(model, x, out_path, **kwargs)
        return True
    except ModuleNotFoundError as e:
        missing = e
    except Exception:
        pass                                # 新导出器其它错误 → 直接回退
    try:
        torch.onnx.export(model, x, out_path, dynamo=False, **kwargs)
        return True
    except ModuleNotFoundError as e:
        missing = e
    except Exception as e:
        print(f"[警告] ONNX 导出失败: {e}", file=sys.stderr)
        return False
    print(f"[警告] ONNX 导出需要 pip 包: {missing}（请执行 pip install onnx）",
          file=sys.stderr)
    return False


def run_ensemble(args, items_by_digit, label_order, out_dir):
    """主划分上训练 N 个不同初始化的模型：比较 单模型 / 多数投票 / 权重平均。

    权重平均模型（model_ensemble.pt）尺寸与单模型相同，是 MCU 可部署的
    "轻量集成"；ONNX 导出（model.onnx）供 STM32Cube.AI Studio 适配。
    """
    split_seed = args.seed
    plan = make_plan(args.split, items_by_digit, VAL_N, TEST_N, split_seed)
    samples = {"train": [], "val": [], "test": []}
    for digit, files in items_by_digit.items():
        idx = label_order.index(digit)
        for f in files:
            samples[plan[f]].append((f, idx))
    test_ds = ScanDataset(samples["test"], args.height, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    targets = np.array([y for _, y in samples["test"]])

    best_states, probs_list, single_accs, seeds, val_accs = [], [], [], [], []
    for i in range(args.ensemble):
        model_seed = split_seed + i * 1000
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        random.seed(model_seed)
        print(f"--- 集成模型 {i + 1}/{args.ensemble} (seed={model_seed}) ---")
        best_state, info = train_model(samples["train"], samples["val"],
                                       model_seed, args, label_order,
                                       args.height, verbose=args.verbose)
        model = TinyCNN(in_h=args.height, in_w=8, n_classes=len(label_order))
        model.load_state_dict(best_state)
        probs = predict_probs(model, test_loader)
        acc = float((probs.argmax(1) == targets).mean())
        print(f"  单模型测试 acc {acc:.3f} (验证 acc {info['best_val_acc']:.3f}, "
              f"用时 {info['elapsed_s']}s)")
        best_states.append(best_state)
        probs_list.append(probs)
        single_accs.append(acc)
        val_accs.append(info["best_val_acc"])
        seeds.append(model_seed)
        meta = {"in_h": args.height, "in_w": 8,
                "n_classes": len(label_order), "label_order": label_order,
                "seed": model_seed, "test_acc": acc,
                "best_val_acc": info["best_val_acc"]}
        torch.save({"state_dict": best_state, "meta": meta},
                   os.path.join(out_dir, f"models_seed{model_seed}.pt"))

    # ---- 多数投票（PC 端识别用）----
    vote_preds = majority_vote(probs_list)
    vote_acc = float((vote_preds == targets).mean())
    cm_vote = confusion_matrix(vote_preds, targets, len(label_order))

    # ---- 部署模型：按验证集最优选择单模型（无测试泄漏）----
    # 说明：跨不同随机初始化直接做权重平均会因卷积核置换对称性失效，
    # 因此 MCU 部署目标 = 验证集最优的单个模型。
    deploy_idx = int(np.argmax(val_accs))
    deploy_state = best_states[deploy_idx]
    deploy_seed = seeds[deploy_idx]
    deploy_model = TinyCNN(in_h=args.height, in_w=8,
                           n_classes=len(label_order))
    deploy_model.load_state_dict(deploy_state)
    deploy_acc = single_accs[deploy_idx]
    deploy_probs = predict_probs(deploy_model, test_loader)
    cm_deploy = confusion_matrix(deploy_probs.argmax(1), targets,
                                 len(label_order))
    n_params = sum(p.numel() for p in deploy_model.parameters())

    # ---- 保存产物 ----
    meta = {"in_h": args.height, "in_w": 8, "n_classes": len(label_order),
            "label_order": label_order, "seed": deploy_seed,
            "test_acc": deploy_acc, "best_val_acc": val_accs[deploy_idx],
            "mode": "ensemble_selected"}
    torch.save({"state_dict": deploy_state, "meta": meta},
               os.path.join(out_dir, "model_best.pt"))
    split_name = (f"split_fixed.json" if args.split == "fixed"
                  else f"split_seed{split_seed}.json")
    with open(os.path.join(out_dir, split_name), "w") as f:
        json.dump(split_manifest(plan, items_by_digit), f,
                  ensure_ascii=False, indent=2)
    try:
        plot_confusion(cm_deploy, label_order,
                       os.path.join(out_dir, "confusion_matrix.png"))
    except ImportError:
        pass

    # ---- ONNX 导出（部署模型）----
    onnx_ok = export_onnx(deploy_model, os.path.join(out_dir, "model.onnx"),
                          args.height, 8)
    deploy = memory_analysis(deploy_model, args.height, 8)

    single_mean = float(np.mean(single_accs))
    single_std = float(np.std(single_accs))
    label, msg = verdict(deploy_acc)

    print("\n" + "=" * 60)
    print(f"单模型测试 acc: {single_mean:.3f} ± {single_std:.3f} "
          f"(n={args.ensemble})")
    print(f"多数投票 acc:   {vote_acc:.3f}  ← PC 端识别用")
    print(f"部署模型 acc:   {deploy_acc:.3f}  ← 验证集最优单模型(seed {deploy_seed})")
    print(f"ONNX 导出: {'成功' if onnx_ok else '失败'} → {out_dir}/model.onnx")
    print(f"权重: fp32 {deploy['weights_fp32_kb']}KB / "
          f"int8 {deploy['weights_int8_kb']}KB | "
          f"激活峰值: fp32 {deploy['peak_activation_fp32_kb']}KB / "
          f"int8 {deploy['peak_activation_int8_kb']}KB")
    print(f"判定(基于部署模型): [{label}]")
    print(msg)
    print("=" * 60)

    return {
        "mode": "ensemble",
        "digits": label_order,
        "height": args.height,
        "split": split_descr(args.split, VAL_N, TEST_N),
        "n_total": sum(len(v) for v in items_by_digit.values()),
        "n_per_class": {d: len(items_by_digit[d]) for d in label_order},
        "primary_seed": split_seed,
        "n_params": n_params,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "ensemble": {
            "n_models": args.ensemble,
            "model_seeds": seeds,
            "single_test_accs": single_accs,
            "single_val_accs": val_accs,
            "single_mean": single_mean,
            "single_std": single_std,
            "vote_test_acc": vote_acc,
            "confusion_vote": cm_vote.tolist(),
            "deploy": {"seed": deploy_seed, "test_acc": deploy_acc,
                       "val_acc": val_accs[deploy_idx],
                       "confusion": cm_deploy.tolist()},
        },
        "deployment": deploy,
        "onnx_exported": onnx_ok,
        "verdict": {"label": label, "message": msg},
    }


# ================================================================ 绘图
def plot_curves(log, out_path):
    import matplotlib
    import matplotlib.pyplot as plt
    setup_cjk_font()
    epochs = [r[0] for r in log]
    tr_loss = [r[1] for r in log]
    va_loss = [r[3] for r in log]
    tr_acc = [r[2] for r in log]
    va_acc = [r[4] for r in log]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, tr_acc, label="train")
    axes[1].plot(epochs, va_acc, label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, label_order, out_path):
    import matplotlib
    import matplotlib.pyplot as plt
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_order)))
    ax.set_yticks(range(len(label_order)))
    ax.set_xticklabels(label_order)
    ax.set_yticklabels(label_order)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================ 判定
def verdict(test_acc):
    if test_acc >= 0.90:
        return ("可行", "测试准确率 ≥90%，该扫描方案足以支撑 0~9 识别，"
                "可继续扩充数据集并推进 PC 实时识别演示。")
    if test_acc >= 0.70:
        return ("基本可行，需改进", "测试准确率 70~90%：方法本身有效，但受"
                "人工扫描不统一影响，建议增加数据量、固定扫描距离/速度。")
    return ("需改进后重新验证", "测试准确率 <70%：当前扫描数据质量或数量不足，"
            "建议改进机械定位与采集质量后再训练验证。")


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description="TCRT5000 数字识别训练与可行性评估")
    ap.add_argument("--src", default="datasets", help="数据集根目录（默认 datasets/）")
    ap.add_argument("--out-dir", default="outputs", help="产物输出目录（默认 outputs/）")
    ap.add_argument("--height", type=int, default=32, help="重采样高度 H（默认 32）")
    ap.add_argument("--seed", type=int, default=42, help="划分/训练随机种子（默认 42）")
    ap.add_argument("--repeats", type=int, default=1,
                    help="用不同 seed 重复实验次数（默认 1，求测试准确率 mean±std）")
    ap.add_argument("--epochs", type=int, default=300, help="最大训练轮数（默认 300）")
    ap.add_argument("--patience", type=int, default=40, help="早停耐心（默认 40）")
    ap.add_argument("--batch", type=int, default=16, help="批大小（默认 16）")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率（默认 1e-3）")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Label Smoothing 系数，0=关闭（默认 0.1，训练期特性）")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW 权重衰减，0=关闭（默认 1e-4，训练期特性）")
    ap.add_argument("--ensemble", type=int, default=0,
                    help="多 seed 投票集成数：>0 时在主划分上训练 N 个模型，"
                         "比较单模型/多数投票/权重平均并导出 ONNX（默认 0=关）")
    ap.add_argument("--split", choices=["random", "fixed"], default="fixed",
                    help="数据划分方式：fixed=按文件名序号固定划分"
                         "（训练1-14+21-26/验证15-17+27-28/测试18-20+29-30，默认）；"
                         "random=每类随机 14/3/3 划分（用 --seed 复现）")
    ap.add_argument("--verbose", action="store_true", help="每 10 轮打印一次指标")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out_dir, ".mplcache"))

    items = scan_sources(args.src)          # [(digit, path), ...]
    if not items:
        sys.exit(f"目录 {args.src} 下没有找到 <digit>_scan/*.csv 数据")
    items_by_digit = defaultdict(list)
    for digit, f in items:
        items_by_digit[digit].append(f)
    label_order = sorted(items_by_digit)    # 按数据实际出现的数字排序，如 0~9
    print(f"数据: {len(items)} 个 CSV（{'/'.join(label_order)}）")
    print(f"划分: {split_descr(args.split, VAL_N, TEST_N)}"
          + ("（0~9 每类 30 个：01~20 MNIST 打印体 + 21~30 字体版）"
             if args.split == "fixed" else ""))
    print(f"模型: TinyCNN H={args.height}×8 通道；"
          f"训练 {args.epochs} epoch 上限，早停 patience={args.patience}"
          f"，Label Smoothing={args.label_smoothing}，"
          f"weight decay={args.weight_decay}")

    if args.ensemble > 0:
        # ---- 多 seed 集成模式：投票 + 权重平均 + ONNX/内存分析 ----
        summary = run_ensemble(args, items_by_digit, label_order, out_dir)
        with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n产物: models_seed*.pt / model_best.pt / model_ensemble.pt / "
              f"model.onnx / train_summary.json → {out_dir}/")
        return

    # ---- 常规模式：多划分稳健性实验 ----
    results = []
    for i in range(args.repeats):
        seed = args.seed + i * 1000
        print(f"--- 实验 {i + 1}/{args.repeats} (seed={seed}) ---")
        r = run_experiment(seed, args, items_by_digit, label_order,
                           out_dir, save_artifacts=(i == 0))
        results.append(r)
        print(f"  完成: 最佳验证 acc {r['best_val_acc']:.3f}"
              f" (epoch {r['best_epoch']})，测试 acc {r['test_acc']:.3f}，"
              f"用时 {r['elapsed_s']}s")

    # ---- 汇总 ----
    pri = results[0]
    test_accs = [r["test_acc"] for r in results]
    summary = {
        "mode": "repeats",
        "digits": label_order,
        "height": args.height,
        "split": split_descr(args.split, VAL_N, TEST_N),
        "n_total": len(items),
        "n_per_class": {d: len(items_by_digit[d]) for d in label_order},
        "repeats": args.repeats,
        "primary_seed": args.seed,
        "n_params": pri["n_params"],
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "test_acc_all_repeats": test_accs,
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "primary": pri,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 判定与打印 ----
    mean_acc = summary["test_acc_mean"]
    label, msg = verdict(mean_acc)
    print("\n" + "=" * 60)
    print(f"测试集准确率: mean={mean_acc:.3f} ± {summary['test_acc_std']:.3f}"
          f" (n={len(test_accs)})")
    print(f"每类准确率(主实验): "
          + ", ".join(f"{d}:{pri['per_class_acc'][d]:.2f}"
                      for d in label_order))
    print(f"混淆矩阵(主实验):\n{np.array(pri['confusion'])}")
    print(f"\n判定: [{label}]")
    print(msg)
    print("=" * 60)
    print(f"产物已保存到 {out_dir}/: model_best.pt, model_final.pt, "
          f"training_log.csv, curves.png, confusion_matrix.png, "
          f"{'split_fixed.json' if args.split == 'fixed' else 'split_seed' + str(args.seed) + '.json'}, "
          f"train_summary.json")


if __name__ == "__main__":
    main()
