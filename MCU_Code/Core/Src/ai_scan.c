/**
  ******************************************************************************
  * @file    ai_scan.c
  * @author  user
  * @brief   On-board AI scan recognition module (ST Edge AI int8 network).
  *
  *  流程（与 PC 端 tools/preprocess.py 完全一致）：
  *    扫描缓冲 → 无纸检测(连续 5 行全高=扫描结束) → 黑边 v2 裁剪
  *    → 线性重采样 32 行 → p1/p99 归一化 → int8 量化(scale≈1/127)
  *    → stai_network_run → 反量化+softmax → 串口输出各数字概率
  *    → LED 数码管：置信度 ≥30% 显示预测数字并保持 5s，否则熄灭。
  ******************************************************************************
  */
#include "ai_scan.h"
#include "network.h"
#include "seg_display.h"
#include "usart.h"          /* huart1 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* 8 路传感器（与 main.c 的 ADC_CH_COUNT 一致） */
#define AI_ADC_CH     8u

/* 无纸判定与扫描缓冲 */
#define AI_IDLE_THR   (4095 - 100)   /* 无纸判定：全部通道 >= 此值 */
#define AI_IDLE_ROWS  5              /* 连续无纸行数 = 一次扫描结束 */
#define AI_MAX_ROWS   256            /* 单次扫描缓冲最大行数 */
#define AI_NP_THR     3995           /* 黑边裁剪：行内任一通道 >= 此值 = 无纸行 */

/* 数码管显示 */
#define SEG_HOLD_MS   5000u          /* 显示保持时间 */
#define SEG_MIN_PROB  0.30f          /* 最低置信度（< 30% 不显示） */

/* ------------------------------------------------------------------ 内部状态 */
static uint16_t scan_buf[AI_MAX_ROWS][AI_ADC_CH];
static uint16_t scan_len = 0;
static uint8_t  in_scan = 0;
static uint8_t  idle_run = 0;

/* ST Edge AI 网络上下文 / 激活缓冲 / IO 指针（模型：int8 1×32×8 → int8 1×10） */
STAI_NETWORK_CONTEXT_DECLARE(ai_ctx, STAI_NETWORK_CONTEXT_SIZE);
STAI_ALIGNED(32) static uint8_t ai_act[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];
static stai_ptr ai_in = NULL;    /* 输入 (1,32,8) int8，256B */
static stai_ptr ai_out = NULL;   /* 输出 (1,10) int8，10B */
static uint32_t seg_until = 0;   /* 数码管保持到该 tick（0=熄灭） */

/* ------------------------------------------------------------------ 工具 */
static void ai_uart_print(const char *s)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), 100);
}

/* 输出 10 类概率：反量化(int8×scale) → softmax → prob[10] (0..1) */
static void ai_softmax(const int8_t *out, float *prob)
{
  float logits[10], mx = -1e30f, sum = 0.0f;
  int i;
  for (i = 0; i < 10; i++)
  {
    logits[i] = (float)out[i] * STAI_NETWORK_OUT_1_SCALE;
    if (logits[i] > mx) mx = logits[i];
  }
  for (i = 0; i < 10; i++)
  {
    logits[i] = expf(logits[i] - mx);
    sum += logits[i];
  }
  for (i = 0; i < 10; i++) prob[i] = logits[i] / sum;
}

static int ai_cmp_float(const void *a, const void *b)
{
  float x = *(const float *)a - *(const float *)b;
  return (x > 0.0f) - (x < 0.0f);
}

/* ------------------------------------------------------------------ 推理 */
/* 对一次完整扫描执行：黑边裁剪 → 重采样 32 行 → 归一化 → int8 量化 → 推理 → 概率串口输出。
   预处理与 PC 端 tools/preprocess.py 完全一致（黑边 v2 / np.interp / p1-p99 归一化）。 */
static void run_scan_inference(void)
{
  uint16_t s = 0, e = scan_len;
  uint32_t k;
  int r, c, i;

  /* 1) 黑边/楔形黑边裁剪：去掉首/尾"含无纸电平(>=3995)通道"的连续行 */
  while (s < e)
  {
    uint8_t np = 0;
    for (k = 0; k < AI_ADC_CH; k++)
      if (scan_buf[s][k] >= AI_NP_THR) { np = 1; break; }
    if (!np) break;
    s++;
  }
  while (e > s)
  {
    uint8_t np = 0;
    for (k = 0; k < AI_ADC_CH; k++)
      if (scan_buf[e - 1][k] >= AI_NP_THR) { np = 1; break; }
    if (!np) break;
    e--;
  }
  uint16_t n = e - s;
  if (n < 4)
  {
    ai_uart_print("\r\n[AI] scan too short, skip\r\n");
    return;
  }

  /* 2) Y 向线性重采样到 32 行（与 PC 端 np.interp 一致） */
  float in32[32][AI_ADC_CH];
  for (r = 0; r < 32; r++)
  {
    float pos = (float)r * (float)(n - 1) / 31.0f;
    uint16_t i0 = (uint16_t)pos;
    uint16_t i1 = (i0 + 1 < n) ? (uint16_t)(i0 + 1) : (uint16_t)(n - 1);
    float f = pos - (float)i0;
    for (c = 0; c < AI_ADC_CH; c++)
      in32[r][c] = (1.0f - f) * (float)scan_buf[s + i0][c] + f * (float)scan_buf[s + i1][c];
  }

  /* 3) 逐样本归一化：p1/p99 线性插值裁剪 → [0,1]（与 PC 端 np.percentile 一致） */
  float vals[32 * 8];
  for (r = 0; r < 32; r++)
    for (c = 0; c < 8; c++) vals[r * 8 + c] = in32[r][c];
  qsort(vals, 32 * 8, sizeof(float), ai_cmp_float);
  {
    float idx1 = (float)(32 * 8 - 1) * 0.01f;
    float idx99 = (float)(32 * 8 - 1) * 0.99f;
    int i1i = (int)idx1, i99i = (int)idx99;
    float lo = vals[i1i] + (idx1 - (float)i1i) * (vals[i1i + 1] - vals[i1i]);
    float hi = vals[i99i] + (idx99 - (float)i99i) * (vals[i99i + 1] - vals[i99i]);
    if (hi <= lo) hi = lo + 1e-6f;
    float gain = 127.0f / (hi - lo);

    /* 4) int8 量化：q = round((v-lo)/(hi-lo)*127)，scale≈1/127、zp=0 */
    for (r = 0; r < 32; r++)
      for (c = 0; c < 8; c++)
      {
        float v = (in32[r][c] - lo) * gain;
        if (v < 0.0f) v = 0.0f;
        if (v > 127.0f) v = 127.0f;
        ai_in[r * 8 + c] = (int8_t)(int)(v + 0.5f);
      }
  }

  /* 5) 推理 */
  stai_return_code rc = stai_network_run((stai_network *)ai_ctx, STAI_MODE_SYNC);
  if (rc != STAI_SUCCESS)
  {
    char err[48];
    sprintf(err, "\r\n[AI] run failed: %d\r\n", (int)rc);
    ai_uart_print(err);
    return;
  }

  /* 6) softmax 概率 */
  float prob[10];
  ai_softmax((const int8_t *)ai_out, prob);

  /* 7) 串口输出各数字预测概率（整数百分比） */
  {
    char line[160];
    int o = 0, best = 0;
    for (i = 1; i < 10; i++)
      if (prob[i] > prob[best]) best = i;
    o += sprintf(line + o, "\r\n[AI] rows=%u best=%d:%d%% |",
                 scan_len, best, (int)(prob[best] * 100.0f + 0.5f));
    for (i = 0; i < 10; i++)
      o += sprintf(line + o, " %d:%d%%", i, (int)(prob[i] * 100.0f + 0.5f));
    o += sprintf(line + o, "\r\n");
    ai_uart_print(line);
  }

  /* 8) 数码管显示：置信度 >= 30% 显示预测数字并保持 5s，否则熄灭 */
  {
    int best = 0;
    for (i = 1; i < 10; i++)
      if (prob[i] > prob[best]) best = i;
    if (prob[best] >= SEG_MIN_PROB)
    {
      SEG_Display_Number((uint8_t)best);
      seg_until = HAL_GetTick() + SEG_HOLD_MS;
    }
    else
    {
      SEG_Display_Clear();
      seg_until = 0;
    }
  }
}

/* ------------------------------------------------------------------ 对外接口 */
/* 初始化 ST Edge AI 网络与 LED 数码管（启动时调用一次） */
void AI_Scan_Init(void)
{
  stai_size in_len = 0, out_len = 0;
  stai_runtime_init();
  stai_network_init((stai_network *)ai_ctx);
  stai_network_set_activations((stai_network *)ai_ctx,
                               (stai_ptr[]){ (stai_ptr)ai_act },
                               STAI_NETWORK_ACTIVATIONS_NUM);
  stai_network_get_inputs((stai_network *)ai_ctx, (stai_ptr *)&ai_in, &in_len);
  stai_network_get_outputs((stai_network *)ai_ctx, (stai_ptr *)&ai_out, &out_len);
  SEG_Display_IO_Init();
  SEG_Display_Init();
  ai_uart_print("\r\n[AI] network ready\r\n");
}

/* 每来一行 8 通道数据调用一次：无纸/扫描结束检测 + 缓冲 + 推理触发 */
void AI_Scan_FeedRow(const uint16_t *row8)
{
  uint32_t k;
  uint8_t idle = 1;

  /* 当前行是否"无纸"（全部通道 >= 4095-100） */
  for (k = 0; k < AI_ADC_CH; k++)
    if (row8[k] < AI_IDLE_THR) { idle = 0; break; }

  if (idle)
  {
    if (in_scan)
    {
      idle_run++;
      if (idle_run >= AI_IDLE_ROWS)
      {
        /* 一次扫描结束 → 板载推理并串口输出各数字概率 */
        run_scan_inference();
        in_scan = 0;
        idle_run = 0;
        scan_len = 0;
      }
    }
  }
  else
  {
    if (!in_scan)
    {
      in_scan = 1;
      idle_run = 0;
      scan_len = 0;
    }
    if (scan_len < AI_MAX_ROWS)
    {
      for (k = 0; k < AI_ADC_CH; k++)
        scan_buf[scan_len][k] = row8[k];
      scan_len++;
    }
    else
    {
      ai_uart_print("\r\n[AI] buffer overflow, reset\r\n");
      in_scan = 0;
      idle_run = 0;
      scan_len = 0;
    }
  }
}

/* 主循环周期调用：数码管显示到时（5s）自动熄灭 */
void AI_Scan_Tick(void)
{
  if (seg_until && (int32_t)(HAL_GetTick() - seg_until) >= 0)
  {
    SEG_Display_Clear();
    seg_until = 0;
  }
}
