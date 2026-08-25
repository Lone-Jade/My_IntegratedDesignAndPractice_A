# 扫描部分（已完成）— AI_TCRT5000_ScanData

扫描数据采集阶段快照：TIM6 定时器分时驱动 + ADC 单通道转换 + USART1 二进制帧上送。

- `Core/`：采集固件（main.c 分时驱动/ADC/组帧；tim/adc/usart/gpio 由 CubeMX 生成）
- `Drivers/ cmake/ *.ioc *.ld startup`：CubeMX 工程与构建
- `docs/`：任务书、电路连接说明

> 本目录为"扫描部分"里程碑快照；含板载 AI 推理与数码管显示的当前固件见 `../MCU_Code/`；
> 采集/图像工具见 `../Python_Tools/`。
