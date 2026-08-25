# 单片机代码（MCU_Code）— 当前完整固件，可构建

板载 AI 识别固件：TIM6 分时驱动采集 + ST Edge AI int8 板载推理 + LED 数码管显示。

## 结构
- `Core/Src/main.c`：采样主循环（分时驱动 + ADC + 二进制帧上送，调用各模块）
- `Core/Src/ai_scan.c`：板载 AI 模块（扫描缓冲/结束检测、预处理、推理、softmax、
  串口输出各数字概率、数码管驱动：置信≥30% 显示 5s）
- `Core/Src/seg_display.c`：单 LED 数码管驱动（PB0~PB7 共阴）
- `Network/`：ST Edge AI 生成的 int8 网络（权重 22.4KB）
- `Middlewares/ST/AI/`：ST Edge AI 运行时（头文件 + CM0+ 运行库）
- `ref/AI_TRY_SEG`：数码管参考工程
- `Drivers/ cmake/ *.ioc *.ld startup`：CubeMX 工程与构建

## 构建与烧录（Linux）
```bash
cmake --preset Debug && cmake --build build/Debug
openocd -f interface/stlink.cfg -f target/stm32g0x.cfg \
        -c "program build/Debug/AI_TRY_TCRT5000.elf verify reset exit"
```

## 串口输出
- 每行原始二进制帧：`AA 55 | 行号u16 | 8×u16 | 校验和`（21B）
- 每次扫描结束输出：`[AI] rows=56 best=9:94% | 0:0% ... 9:94%`
- 资源占用：Flash 53.1%（69.6KB）/ RAM 12.0%（17.8KB）
