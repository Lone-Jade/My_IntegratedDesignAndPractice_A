#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX int8 量化（QDQ 格式），供 STM32Cube.AI Studio / stedgeai 使用
==================================================================
ST Edge AI Core 对 float 输入的 ONNX **不自动做 int8 量化**（直接
`--input-data-type int8` 会报错：input is not quantized）。官方标准流程是
先用 **ONNX Runtime 静态量化**把 float32 ONNX 转成 int8 QDQ 模型
（Conv/Gemm 权重与激活均 int8，图内带 QuantizeLinear/DequantizeLinear
量化参数），再导入 ST Edge AI（Studio 或 stedgeai CLI）。

量化后的权重约 22 KiB（float32 的 1/4），适配 STM32G0B1 的 128KB Flash。

用法:
    pip install onnxruntime onnx            # 仅本脚本需要（量化+验证）
    python3 tools/quantize_onnx.py                            # model_best.onnx → model_best_int8.onnx
    python3 tools/quantize_onnx.py --model outputs/model.onnx --out outputs/model_int8.onnx
    python3 tools/quantize_onnx.py --calib outputs/ai_studio_test.npz --samples 30

校准/验证数据：默认用 outputs/ai_studio_test.npz（x_test 与模型输入
1×1×32×8 对齐、值域 [0,1]；y_test 为 one-hot 真值）。

依赖: numpy, onnx, onnxruntime
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = "outputs/model_best.onnx"
DEFAULT_OUT = "outputs/model_best_int8.onnx"
DEFAULT_CALIB = "outputs/ai_studio_test.npz"


class NpzDataReader:
    """从 .npz 的 x_test 读取校准样本（onnxruntime quantize_static 需要）。"""

    def __init__(self, npz_path, n_samples=0, batch=1):
        data = np.load(npz_path)
        x = np.asarray(data["x_test"], dtype=np.float32)
        if n_samples and n_samples < len(x):
            x = x[:n_samples]
        self._x = x
        self._batch = batch

    def get_next(self):
        if self._iter >= len(self._x):
            return None
        x = self._x[self._iter:self._iter + self._batch]
        self._iter += self._batch
        return {"input": x}

    def rewind(self):
        self._iter = 0


def main():
    ap = argparse.ArgumentParser(
        description="ONNX int8 (QDQ) 静态量化，供 STM32Cube.AI 使用")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="输入 float32 ONNX")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 int8 QDQ ONNX")
    ap.add_argument("--calib", default=DEFAULT_CALIB,
                    help="校准数据集 .npz（x_test）")
    ap.add_argument("--samples", type=int, default=30,
                    help="用于校准/验证的样本数（默认 30，<=0 用全部）")
    ap.add_argument("--per-channel", action="store_true", default=True,
                    help="卷积权重按通道量化（默认开）")
    ap.add_argument("--no-per-channel", dest="per_channel",
                    action="store_false")
    args = ap.parse_args()

    for f in (args.model, args.calib):
        if not os.path.exists(f):
            sys.exit(f"找不到文件: {f}")

    from onnxruntime.quantization import (QuantFormat, QuantType,
                                          quantize_static)

    print(f"校准数据: {args.calib} (最多 {args.samples or '全部'} 个样本)")
    dr = NpzDataReader(args.calib, n_samples=args.samples)
    dr.rewind()

    extra = {"ActivationSymmetric": True, "WeightSymmetric": True}
    quantize_static(
        args.model, args.out, dr,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options=extra,
    )
    size = os.path.getsize(args.out)
    print(f"已生成 int8 QDQ 模型: {args.out} ({size/1024:.1f} KiB)")

    # ---- 用 ONNX Runtime 对比量化前后准确率 ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("提示: 安装 onnxruntime 可对比量化前后准确率")
        return

    d = np.load(args.calib)
    x, y = np.asarray(d["x_test"], np.float32), d["y_test"]
    y_idx = np.argmax(y, axis=-1).reshape(-1) if y.ndim > 1 else y
    if args.samples and args.samples < len(x):
        x, y_idx = x[:args.samples], y_idx[:args.samples]

    def acc(model_path):
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        pred = []
        for i in range(len(x)):
            out = sess.run(None, {inp: x[i:i + 1]})[0]
            pred.append(int(np.argmax(out)))
        return float(np.mean(np.array(pred) == y_idx))

    a0 = acc(args.model)
    a1 = acc(args.out)
    print(f"ONNX Runtime 验证: float32 acc={a0:.3f}  int8 acc={a1:.3f}"
          f"  差={a1 - a0:+.3f} ({len(x)} 样本)")


if __name__ == "__main__":
    main()
