/* USER CODE BEGIN Header */

/**
  ******************************************************************************
  * @file    seg_display.c
  * @brief   One-digit 7-segment display driver.
  *
  *          The display is common-cathode and uses PB0-PB7 as:
  *            PB0=A, PB1=B, PB2=C, PB3=D, PB4=E, PB5=F, PB6=G, PB7=DP
  *
  *          A segment is lit when the corresponding GPIO output is high.
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

/* Includes ------------------------------------------------------------------*/
#include "seg_display.h"

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/* Private define ------------------------------------------------------------*/
#define SEG_ALL_PINS  (SEG_A_Pin | SEG_B_Pin | SEG_C_Pin | SEG_D_Pin | \
                       SEG_E_Pin | SEG_F_Pin | SEF_G_Pin | SEG_DP_Pin)

/* Private variables ---------------------------------------------------------*/
/* Active-high codes for common-cathode 7-segment display, DP always off. */
static const uint16_t seg_code[10] =
{
  0x3F, /* 0: A B C D E F */
  0x06, /* 1: B C */
  0x5B, /* 2: A B D E G */
  0x4F, /* 3: A B C D G */
  0x66, /* 4: B C F G */
  0x6D, /* 5: A C D F G */
  0x7D, /* 6: A C D E F G */
  0x07, /* 7: A B C */
  0x7F, /* 8: A B C D E F G */
  0x6F, /* 9: A B C D F G */
};

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/* Exported functions --------------------------------------------------------*/

/**
  * @brief  Initialize the 7-segment display driver.
  * @note   Assumes the GPIO pins have already been configured by MX_GPIO_Init().
  * @retval None
  */
void SEG_Display_Init(void)
{
  SEG_Display_Clear();
}

/**
  * @brief  Display a decimal digit (0-9) on the 7-segment display.
  * @param  number: digit to display. Values out of range turn the display off.
  * @retval None
  */
void SEG_Display_Number(uint8_t number)
{
  uint16_t code;

  if (number < 10u)
  {
    code = seg_code[number];
  }
  else
  {
    code = 0x00u;
  }

  /* First clear all segment pins, then set only the segments needed. */
  HAL_GPIO_WritePin(SEG_A_GPIO_Port, SEG_ALL_PINS, GPIO_PIN_RESET);
  if (code != 0x00u)
  {
    HAL_GPIO_WritePin(SEG_A_GPIO_Port, code, GPIO_PIN_SET);
  }
}

/**
  * @brief  Turn off all segments (including decimal point).
  * @retval None
  */
void SEG_Display_Clear(void)
{
  HAL_GPIO_WritePin(SEG_A_GPIO_Port, SEG_ALL_PINS, GPIO_PIN_RESET);
}

/* USER CODE BEGIN 3 */

/* USER CODE END 3 */
