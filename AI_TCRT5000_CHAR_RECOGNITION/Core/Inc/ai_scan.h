/**
  ******************************************************************************
  * @file    ai_scan.h
  * @author  user
  * @brief   On-board AI scan recognition module
  *
  *          ST Edge AI int8 网络（1×32×8 → 10 类概率）+ 扫描缓冲/结束检测
  *          + LED 数码管显示（置信度 ≥30% 显示预测数字并保持 5s）。
  *          从 main.c 独立出来，main 只需在每行数据就绪时调用 AI_Scan_FeedRow。
  ******************************************************************************
  */
#ifndef AI_SCAN_H
#define AI_SCAN_H

#include "main.h"

/* 模块 API：
   - AI_Scan_Init()    : 初始化 ST Edge AI 网络与 LED 数码管（启动时调用一次）
   - AI_Scan_FeedRow() : 每来一行 8 通道数据调用一次。内部做无纸/扫描结束检测，
                         扫描完成时自动执行推理，串口输出各数字预测概率，
                         并按置信度驱动数码管显示
   - AI_Scan_Tick()    : 主循环周期调用，处理数码管 5s 到时自动熄灭          */
void AI_Scan_Init(void);
void AI_Scan_FeedRow(const uint16_t *row8);
void AI_Scan_Tick(void);

#endif /* AI_SCAN_H */
