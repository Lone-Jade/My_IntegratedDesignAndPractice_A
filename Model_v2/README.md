# 第二版模型（Model_v2）— 最新结果

基于 **300 个扫描样本**（0~9 各 30 个：MNIST 打印体 01~20 + 10 种标准字体 21~30）、
**固定划分（新旧字体混合）** 重训并部署的模型。对应主仓库提交 5553da2 / 56390ac / 314c0c0。

## python/ — 训练与量化脚本
- `train_digit.py`：PyTorch 微型 CNN（2.26 万参数）训练 0~9；`--split fixed` 固定划分 + Label Smoothing/weight decay + 多 seed 集成 + ONNX 导出
- `preprocess.py`：预处理管线（无纸电平黑边 v2 裁剪 / 重采样 32 行 / 逐样本归一化 / 增强）
- `eval_model.py`：用已训练模型评估指定扫描集（新数据复测，如 `--digits 0-4 --seq 21-30`）
- `check_samples.py`：样本质量体检 + 留出误判分析（建议重扫清单）
- `export_ai_studio_data.py`：导出 ST Edge AI 验证 .npz（x_test + one-hot y_test）
- `quantize_onnx.py`：ONNX int8 QDQ 静态量化（权重 22.4KB，零精度损失）

## c/ — C 模型（第二轮 stedgeai generate-2 生成的 int8 网络 + 运行时）
- `Network/`：network.c/h、network_data.c/h、network_details.h（2026-08-27 生成，权重 22,928B）
- `Middlewares/ST/AI/`：ST Edge AI 运行时（头文件 + Cortex-M0+ 运行库）

## artifacts/ — 模型产物、数据与图片
- **模型**：`model_best.pt`（验证集最优部署模型，seed 1042）、`model_final.pt`、`models_seed{42,1042,2042,3042,4042}.pt`（5 个集成模型）
- **ONNX**：`model.onnx`（float32）、`model_best.onnx`、`model_best_int8.onnx`（int8 QDQ，供 ST Edge AI 导入）
- **数据**：`ai_studio_test.npz`（验证数据）、`training_log.csv`（每 epoch 过程数据）、`split_fixed.json`（固定划分清单）、`summary_fixed_repeats.json` / `summary_fixed_ensemble.json` / `train_summary.json`
- **图片**：`curves.png`（训练曲线）、`confusion_matrix.png`（50 样本测试混淆矩阵）、`preprocess_grid.png`（预处理管线预览）
- **报告**：`识别可行性测试报告.md`（完整报告，含第一轮 §1~§9 与第二轮 §10~§11 结果与提升分析）

## 结果速览（固定划分，50 样本测试集）
- 5 seed 测试准确率：80.0 / 82.0 / 82.0 / 80.0 / 82.0 → **81.2% ± 1.0%**；多 seed 多数投票 **88.0%**；
- 部署模型（验证集最优 seed 1042，无测试泄漏）测试 **82.0%**（41/50）；
- 新字体（域外）识别：旧模型 16%/52% → **94%**；
- int8 部署：权重 22,928B（22.4KB）、激活 8,192B（8KB）、macc 639,682；`stedgeai validate` 50 样本 **82.0%**，C 模型 vs 参考 **rmse=0、nse=1.0、cos=1.0**（零精度损失）；
- 固件集成后 Flash **53.10%** / RAM **12.04%**（见 `../MCU_Code/`）。

## 与第一版的对比

| 指标 | Model_v1（200 样本，随机 14/3/3） | Model_v2（300 样本，固定 20/5/5） |
| --- | --- | --- |
| 测试集规模 | 30 样本 | 50 样本 |
| 单模型测试均值 | 81.3% ± 6.5% | 81.2% ± 1.0% |
| 多数投票（PC 端） | 83.3% | **88.0%** |
| 部署模型测试 | 76.67% | **82.0%** |
| 新字体（域外）识别 | 16% / 52% | **94%** |
| int8 部署精度损失 | 零 | 零 |

## 复现
```bash
python3 python/train_digit.py --split fixed --repeats 5     # 固定划分重复实验
python3 python/train_digit.py --split fixed --ensemble 5    # 多 seed 投票 + ONNX 导出（部署模型）
python3 python/export_ai_studio_data.py                     # 生成 ai_studio_test.npz
python3 python/quantize_onnx.py                             # int8 量化 → model_best_int8.onnx
```
