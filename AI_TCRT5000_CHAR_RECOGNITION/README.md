# AI_TCRT5000_CHAR_RECOGNITION — 最新完整固件（板载 AI 识别），可构建

板载 AI 识别固件：TIM6 分时驱动采集 + ST Edge AI int8 板载推理（第二轮 generate-2 模型）+ LED 数码管显示。

## 结构
- `Core/Src/main.c`：采样主循环（TIM6 分时驱动 + ADC 采集 + 二进制帧上送，调用各模块）
- `Core/Src/ai_scan.c`：板载 AI 模块（扫描缓冲/结束检测、预处理、`stai_network_run` 推理、
  softmax、串口输出各数字概率、数码管驱动：置信度≥30% 显示预测数字并保持 5s）
- `Core/Src/seg_display.c`：单 LED 数码管驱动（PB0~PB7 共阴）
- `Network/`：ST Edge AI 生成的 int8 网络（2026-08-27 generate-2，权重 22.4KB）
- `Middlewares/ST/AI/`：ST Edge AI 运行时（头文件 + CM0+ 运行库）
- `Drivers/ cmake/ *.ioc *.ld startup .vscode`：CubeMX 工程与构建/调试配置

## 构建与烧录（Linux）
```bash
cmake --preset Debug && cmake --build build/Debug
openocd -f interface/stlink.cfg -f target/stm32g0x.cfg \
        -c "program build/Debug/AI_TCRT5000_CHAR_RECOGNITION.elf verify reset exit"
```

## 串口输出
- 每行原始二进制帧：`AA 55 | 行号u16 | 8×u16 | 校验和`（21B，行间隔 16ms）
- 每次扫描结束输出：`[AI] rows=56 best=9:94% | 0:0% ... 9:94%`
- 资源占用：Flash 53.10%（约 68KB）/ RAM 12.04%（约 17.3KB）

## 模型更新说明
- 本固件已集成**第二轮** int8 模型：`Network/network_data.c`（2026-08-27 generate-2，
  `stedgeai validate` 50 样本 **82.0%**、与参考模型 rmse=0/cos=1.0 零精度损失）；
- 更换模型只需替换 `Network/` 下全部 5 个文件（`network.c/h`、`network_data.c/h`、`network_details.h`），`ai_scan.c` 无需改动（API 不变，
  输出反量化 scale 由 `network.h` 的 `STAI_NETWORK_OUT_1_SCALE` 宏自动更新）；
- 模型训练/量化工具与验证数据见 `../Model_v2/`。
