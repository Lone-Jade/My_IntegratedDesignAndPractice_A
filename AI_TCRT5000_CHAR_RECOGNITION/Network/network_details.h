/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-08-27T12:41:52+0800
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[10]) {
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 8, 1}}, .scale = {1, (const float[1]){0.007874015718698502}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "input_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 8, 8}}, .scale = {1, (const float[1]){0.013184669427573681}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_features_features_0_Conv_output_0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 8, 8}}, .scale = {1, (const float[1]){0.013184669427573681}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_features_features_1_Relu_output_0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 4, 16}}, .scale = {1, (const float[1]){0.034010473638772964}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_features_features_2_Conv_output_0_output" },
   { .size_bytes = 1728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 18, 6, 16}}, .scale = {1, (const float[1]){0.034010473638772964}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_features_features_5_Conv_output_0_pad_before_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 2, 32}}, .scale = {1, (const float[1]){0.05437971651554108}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_features_features_5_Conv_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 8, 2}}, .scale = {1, (const float[1]){0.05437971651554108}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_head_head_0_Flatten_output_0_to_chlast_output" },
   { .size_bytes = 32, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {1, (const float[1]){0.07400786131620407}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_head_head_1_Gemm_output_0_output" },
   { .size_bytes = 32, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {1, (const float[1]){0.07400786131620407}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "_head_head_2_Relu_output_0_output" },
   { .size_bytes = 10, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 10}}, .scale = {1, (const float[1]){0.044995855540037155}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "output_QuantizeLinear_Input_output" }
  },
  .nodes = (const stai_node_details[9]){
    {.id = 13, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _features_features_0_Conv_output_0 */
    {.id = 16, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _features_features_1_Relu_output_0 */
    {.id = 25, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _features_features_2_Conv_output_0 */
    {.id = 28, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _features_features_5_Conv_output_0_pad_before */
    {.id = 34, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _features_features_5_Conv_output_0 */
    {.id = 37, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _head_head_0_Flatten_output_0_to_chlast */
    {.id = 40, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _head_head_1_Gemm_output_0 */
    {.id = 43, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* _head_head_2_Relu_output_0 */
    {.id = 46, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} } /* output_QuantizeLinear_Input */
  },
  .n_nodes = 9
};
#endif

