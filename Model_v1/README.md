# 第一版模型（Model_v1）— 200 样本、随机 14/3/3

## python/ — 训练与量化脚本
- `train_digit.py`：PyTorch 微型 CNN（22.6k 参数）训练 0~9；14/3/3 划分、Label Smoothing、
  weight decay、多 seed 集成（`--ensemble`）、ONNX 导出、部署内存分析
- `preprocess.py`：预处理管线（无纸电平黑边 v2 裁剪 / 重采样 32 行 / 归一化）
- `quantize_onnx.py`：ONNX int8 QDQ 静态量化（权重 22.4KB，零精度损失）
- `export_ai_studio_data.py`：导出 ST Edge AI 验证 .npz（x_test + y_test one-hot）
- `check_samples.py`：样本质量体检 + 5-seed 留出误判分析

## c/ — C 模型（ST Edge AI 生成的 int8 网络 + 运行时）
- `Network/`：network.c/h、network_data.c/h、network_details.h（int8 QDQ 模型）
- `Middlewares/ST/AI/`：运行时头文件 + Cortex-M0+ 运行库（NetworkRuntime1201）

## artifacts/ — 模型产物与报告
- `model_best.pt`（验证集最优部署模型，seed 1042）、`model_final.pt`
- `model.onnx`（float32）、`model_best_int8.onnx`（int8 QDQ，供 ST Edge AI 导入）
- `ai_studio_test.npz`（验证数据）、`summary_{baseline,enhanced,ensemble}.json`、`training_log.csv`
- `curves.png`（训练曲线，由 `training_log.csv` 绘制）、`confusion_matrix.png`（部署模型 30 样本测试混淆矩阵）
- `识别可行性测试报告.md`（第一轮完整测试报告：数据/预处理/增强/模型/结果/部署分析）

## 结果速览
- 基线 75.3%±4.0% → LS+WD 81.3%±6.5%（5 划分）；多 seed 投票 83.3%；
- 部署模型（seed 1042）测试 76.67%（30 样本），int8 与 float32 均 76.67%（零损失）。

> 最新版模型见 `../Model_v2/`（300 样本、固定划分，部署模型测试 82.0%、新字体识别 94%）。
