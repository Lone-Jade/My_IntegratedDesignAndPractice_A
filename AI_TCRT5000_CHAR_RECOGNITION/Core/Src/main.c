/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include "main.h"
#include "adc.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ai_scan.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#include <stdio.h>
#include <string.h>
#include "tim.h"

#define ADC_CH_COUNT  8u   /* 8 路传感器 */

/* LED 控制脚映射：CH0..CH7 -> PD0..PD6、PD8（GPIO_0..GPIO_7，见 main.h） */
static const uint16_t led_pins[ADC_CH_COUNT] = {
  GPIO_0_Pin, GPIO_1_Pin, GPIO_2_Pin, GPIO_3_Pin,
  GPIO_4_Pin, GPIO_5_Pin, GPIO_6_Pin, GPIO_7_Pin
};

/* ADC 通道号映射：CH0..CH7 -> ADC1_IN0..IN7（PA0..PA7），由 CubeMX 配置为模拟输入 */
static const uint32_t adc_ch[ADC_CH_COUNT] = {
  ADC_CHANNEL_0, ADC_CHANNEL_1, ADC_CHANNEL_2, ADC_CHANNEL_3,
  ADC_CHANNEL_4, ADC_CHANNEL_5, ADC_CHANNEL_6, ADC_CHANNEL_7
};

/* 一行数据缓冲、行就绪标志、行号、当前采样通道 */
static uint16_t adc_row[ADC_CH_COUNT];
static volatile uint32_t row_ready = 0;
static uint16_t row_cnt = 0;
static uint32_t step = 0;

/* 通过 USART1 发送字符串 */
static void uart_print(const char *s)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), 100);
}

/* 发送一行二进制帧：AA 55 | 行号u16(LE) | 8×u16(LE) | 校验和u8 */
static void send_row_frame(void)
{
  uint8_t frame[2 + 2 + 2 * ADC_CH_COUNT + 1];
  uint8_t csum = 0;
  frame[0] = 0xAA;
  frame[1] = 0x55;
  frame[2] = (uint8_t)(row_cnt & 0xFFu);
  frame[3] = (uint8_t)((row_cnt >> 8) & 0xFFu);
  for (uint32_t i = 0; i < ADC_CH_COUNT; i++)
  {
    frame[4 + 2 * i]     = (uint8_t)(adc_row[i] & 0xFFu);
    frame[4 + 2 * i + 1] = (uint8_t)((adc_row[i] >> 8) & 0xFFu);
  }
  for (uint32_t i = 2; i < sizeof(frame) - 1; i++)
  {
    csum += frame[i];
  }
  frame[sizeof(frame) - 1] = csum;
  HAL_UART_Transmit(&huart1, frame, sizeof(frame), 100);
}

/* TIM6 更新中断回调（CubeMX 生成的 TIM6_DAC_LPTIM1_IRQHandler -> HAL_TIM_IRQHandler 调用）：
   分时驱动——每次只点亮一路发射管，测其对应通道，然后切下一路。
   LED[step] 自上一拍起已点亮约一个周期(TIM6 周期)，采样时已稳定。 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim == &htim6)
  {
    uint32_t n = step;
    ADC_ChannelConfTypeDef sConfig = {0};

    /* 1. 用 HAL 把 rank1 配置为当前通道（单次转换，不存在扫描溢出问题） */
    sConfig.Channel = adc_ch[n];
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLINGTIME_COMMON_1;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);

    /* 2. 单次转换 + 读取 */
    if (HAL_ADC_Start(&hadc1) == HAL_OK)
    {
      if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
      {
        adc_row[n] = (uint16_t)HAL_ADC_GetValue(&hadc1);
      }
      else
      {
        adc_row[n] = 0xFFFFu;   /* 超时标记，便于发现故障 */
      }
      HAL_ADC_Stop(&hadc1);
    }

    /* 3. 关当前发射管，点亮下一路（下一拍采样它时已稳定） */
    HAL_GPIO_WritePin(GPIOD, led_pins[n], GPIO_PIN_RESET);
    step = (n + 1) & 7u;
    HAL_GPIO_WritePin(GPIOD, led_pins[step], GPIO_PIN_SET);

    /* 4. 8 路采完 = 一行完成 */
    if (n == 7u)
    {
      row_cnt++;
      row_ready = 1;
    }
  }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_USART1_UART_Init();
  MX_TIM6_Init();
  /* USER CODE BEGIN 2 */
uart_print("\r\n=== TCRT5000 TIM6 time-division scan ===\r\n");

  /* TIM6 初始化（CubeMX 配置生成；你在 CubeMX 配好 TIM6 并 Generate 后，
     这里会多一次自动生成的调用，重复调用无害） */
  MX_TIM6_Init();

  /* 启动 TIM6（周期在 CubeMX 的 TIM6 参数里配置，当前 2ms）。
     先点亮第 0 路：首个节拍采样它时已稳定。 */
  HAL_GPIO_WritePin(GPIOD, led_pins[0], GPIO_PIN_SET);
  HAL_TIM_Base_Start_IT(&htim6);

  /* 初始化板载 AI 扫描识别模块（int8 网络 + LED 数码管） */
  AI_Scan_Init();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    /* 数码管显示到时（5s）自动熄灭 */
    AI_Scan_Tick();

    /* 有新的行数据：上送二进制帧，并交给 AI 模块做缓冲/扫描结束检测/推理 */
    if (row_ready)
    {
      send_row_frame();
      AI_Scan_FeedRow(adc_row);
      row_ready = 0;
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 8;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
