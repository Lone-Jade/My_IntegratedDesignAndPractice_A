# Python 工具（Python_Tools）

| 工具 | 功能 |
| --- | --- |
| `scan_view.py` | 串口采集 + 自动分段存 CSV |
| `show_scan.py` | CSV 颜色深度图/图像重建 |
| `make_print_sheet.py` | MNIST 打印稿生成（A4 卡片） |
| `check_quality.py` | 扫描 CSV 质量体检（早期版本） |
| `preprocess.py` | 预处理管线（黑边v2/重采样/归一化） |
| `train_digit.py` | PyTorch CNN 训练 0~9 + 集成 + ONNX 导出 |
| `check_samples.py` | 样本质量体检 + 留出误判分析（建议重扫清单） |
| `export_ai_studio_data.py` | 导出 ST Edge AI 验证 .npz |
| `quantize_onnx.py` | ONNX int8 QDQ 量化 |
| `show_prediction.py` | 串口预测结果终端显示（条形图 + 历史） |

依赖：numpy / torch(CPU) / matplotlib / pyserial / onnx / onnxruntime（按需）。
