# Python 工具（Python_Tools）— 最新版

`tools/` 下 12 个工具：

| 工具 | 功能 |
| --- | --- |
| `scan_view.py` | 串口采集 + 自动分段存 CSV（按"数据恢复 4095"分段） |
| `show_scan.py` | CSV 颜色深度图/图像重建（`--thresh` 纯黑白二值化，自动中文字体） |
| `make_print_sheet.py` | MNIST 打印稿生成（反色 + 加粗 + A4 排版；torchvision/gz 双数据源） |
| `make_print_sheet_fonts.py` | **字体版打印稿生成**（10 种系统标准字体，编号 x-21~x-30） |
| `check_quality.py` | 扫描 CSV 质量体检（早期版本） |
| `preprocess.py` | 预处理管线（无纸电平黑边 v2 裁剪 / 重采样 32 行 / 归一化 / 增强） |
| `train_digit.py` | PyTorch CNN 训练 0~9（固定划分 + Label Smoothing/weight decay + 集成 + ONNX 导出） |
| `eval_model.py` | 用已训练模型评估指定扫描集（新数据复测，如 `--digits 0-4 --seq 21-30`） |
| `check_samples.py` | 样本质量体检 + 留出误判分析（建议重扫清单） |
| `export_ai_studio_data.py` | 导出 ST Edge AI 验证 .npz（x_test + one-hot y_test） |
| `quantize_onnx.py` | ONNX int8 QDQ 静态量化（→ model_best_int8.onnx） |
| `show_prediction.py` | 串口预测结果终端显示（条形图 + 历史） |

依赖：numpy / torch(CPU) / matplotlib / pyserial / onnx / onnxruntime（按需）。

## 常用示例
```bash
python3 tools/train_digit.py --split fixed --repeats 5      # 固定划分重复实验（默认 fixed）
python3 tools/train_digit.py --split fixed --ensemble 5     # 多 seed 投票 + ONNX 导出（部署模型）
python3 tools/eval_model.py --digits 0-4 --seq 21-30        # 用已保存模型评估新字体样本（复测）
python3 tools/export_ai_studio_data.py                      # 生成 ST Edge AI 验证 .npz
python3 tools/quantize_onnx.py                              # int8 量化
python3 tools/show_prediction.py /dev/ttyUSB0               # 板载推理结果终端显示
python3 tools/scan_view.py /dev/ttyUSB0 --image             # 采集 + 自动分段
python3 tools/make_print_sheet_fonts.py                     # 生成字体版打印稿
```
