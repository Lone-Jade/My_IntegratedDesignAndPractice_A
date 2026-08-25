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

/* Exported functions prototypes ---------------------------------------------*/
void SEG_Display_Init(void);
void SEG_Display_Number(uint8_t number);
void SEG_Display_Clear(void);

#ifdef __cplusplus
}
#endif

#endif /* __SEG_DISPLAY_H */
