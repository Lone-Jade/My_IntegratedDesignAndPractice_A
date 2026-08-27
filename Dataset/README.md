# 数据集（Dataset）— 最新版

`datasets/<digit>_scan/<digit>-NN.csv`：数字 0~9 各 30 个，共 **300 个扫描样本**。

- 每文件 = 一张独立纸片的一次扫描（8 通道 ADC 原始值，0~4095；白纸=低值、黑墨=高值）；
- 命名：`<digit>-<NN>.csv`；NN = **01~20** 为 MNIST 打印体，**21~30** 为 10 种标准字体
  （`Python_Tools/tools/make_print_sheet_fonts.py` 生成的 DejaVu/Liberation/URW/C059 等）；
- 行列格式：`row,ch0,ch1,...,ch7`；行数 22~118 不等（人工扫描快慢不同）；
- **固定划分**（新旧字体按比例混入各集合）：训练 = 1-14 + 21-26、验证 = 15-17 + 27-28、
  测试 = 18-20 + 29-30（每类 20/5/5），划分清单见 `Model_v2/artifacts/split_fixed.json`；
- 预处理：`Model_v2/python/preprocess.py`（无纸电平黑边 v2 裁剪 → 重采样 32 行 → 归一化）；
- 质量：`Python_Tools/tools/check_samples.py` 可复检（黑边/墨量/截断/留出误判）。
