/* USER CODE BEGIN Header */

/**
  ******************************************************************************
  * @file    seg_display.h
  * @brief   This file contains the common defines and prototypes for the
  *          one-digit 7-segment display driver.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __SEG_DISPLAY_H
#define __SEG_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* 引脚定义（共阴单数码管：PB0=A .. PB5=F, PB6=G, PB7=DP）。
   本工程未在 CubeMX 引脚表登记这些脚，故在此定义（重生成不丢失）。 */
#define SEG_A_Pin        GPIO_PIN_0
#define SEG_A_GPIO_Port  GPIOB
#define SEG_B_Pin        GPIO_PIN_1
#define SEG_B_GPIO_Port  GPIOB
#define SEG_C_Pin        GPIO_PIN_2
#define SEG_C_GPIO_Port  GPIOB
#define SEG_D_Pin        GPIO_PIN_3
#define SEG_D_GPIO_Port  GPIOB
#define SEG_E_Pin        GPIO_PIN_4
#define SEG_E_GPIO_Port  GPIOB
#define SEG_F_Pin        GPIO_PIN_5
#define SEG_F_GPIO_Port  GPIOB
#define SEF_G_Pin        GPIO_PIN_6   /* 与 ref 工程一致（保留原名 SEF_G） */
#define SEF_G_GPIO_Port  GPIOB
#define SEG_DP_Pin       GPIO_PIN_7
#define SEG_DP_GPIO_Port GPIOB

/* Exported functions prototypes ---------------------------------------------*/
void SEG_Display_IO_Init(void);      /* 使能 GPIOB 时钟并配置 PB0~PB7 为推挽输出 */
void SEG_Display_Init(void);
void SEG_Display_Number(uint8_t number);
void SEG_Display_Clear(void);

#ifdef __cplusplus
}
#endif

#endif /* __SEG_DISPLAY_H */
