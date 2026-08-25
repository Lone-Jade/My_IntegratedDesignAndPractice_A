# 数据集（Dataset）

`datasets/<digit>_scan/<digit>-NN.csv`：数字 0~9 各 20 个，共 **200 个扫描样本**。

- 每文件 = 一张独立纸片的一次扫描（8 通道 ADC 原始值，0~4095；白纸=低值、黑墨=高值）；
- 行列格式：`row,ch0,ch1,...,ch7`；行数 22~118 不等（人工扫描快慢不同）；
- 采集：`Python_Tools/tools/scan_view.py`；命名：`<digit>-<NN>.csv`；
- 预处理：`Model_v1/python/preprocess.py`（黑边 v2 裁剪 → 重采样 32 行 → 归一化）；
- 质量：`Model_v1/python/check_samples.py` 可复检（黑边/墨量/截断/留出误判）。
