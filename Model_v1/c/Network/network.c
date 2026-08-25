/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-24T22:48:57+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x1bef08bc3e3ab2bfbcfa098848707c82"
#define _STAI_NETWORK_DATETIME            "2026-08-24T22:48:57+0800"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 1),
  .tool_version = STAI_INIT_VERSION(4, 0, 1),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 1, 32, 8),
      STAI_DECLARE_ARRAY(float, 1, 0.007874015718698502f),
      STAI_DECLARE_ARRAY(int16_t, 1, 0)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.058477964252233505f),
      STAI_DECLARE_ARRAY(int16_t, 1, 0)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 8192),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 22928),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013158309273421764f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013158309273421764f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03687465563416481f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_Conv_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020376869942992926f, 0.0026627143379300833f, 0.0019200798124074936f, 0.0021570748649537563f, 0.001694760867394507f, 0.002761394251137972f, 0.0023276770953089f, 0.00187350541818887f, 0.0034388327039778233f, 0.0027463436126708984f, 0.0025568793062120676f, 0.0021067725028842688f, 0.0018712497549131513f, 0.003351427149027586f, 0.00228578457608819f, 0.0024142770562320948f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_Conv_output_0_scratch1_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03687465563416481f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_Conv_output_0_scratch2_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03687465563416481f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_5_Conv_output_0_pad_before_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03687465563416481f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_5_Conv_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07734590023756027f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_5_Conv_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018299899529665709f, 0.0008360672509297729f, 0.0019256839295849204f, 0.0019408277003094554f, 0.0019990582950413227f, 0.0013615958159789443f, 0.0024661601055413485f, 0.0016570007428526878f, 0.0018686800030991435f, 0.0015916803386062384f, 0.0017072203336283565f, 0.0017720661126077175f, 0.0020073449704796076f, 0.0012697408674284816f, 0.0016922010108828545f, 0.002138203475624323f, 0.0016900756163522601f, 0.0014875412452965975f, 0.0013510960852727294f, 0.0014927734155207872f, 0.0013198627857491374f, 0.001734479214064777f, 0.0015356324147433043f, 0.0018304464174434543f, 0.0027579220477491617f, 0.000883104745298624f, 0.0015580567996948957f, 0.0018093115650117397f, 0.0013869066024199128f, 0.0020271041430532932f, 0.0015931264497339725f, 0.0013455951120704412f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_5_Conv_output_0_scratch1_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07734590023756027f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_5_Conv_output_0_scratch2_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07734590023756027f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_head_head_0_Flatten_output_0_to_chlast_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07734590023756027f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_head_head_1_Gemm_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0902070477604866f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_head_head_1_Gemm_output_0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013633720809593797f, 0.0013485484523698688f, 0.00045563929597847164f, 0.0015558097511529922f, 0.00044766292558051646f, 0.0004883044166490436f, 0.00038992249756120145f, 0.0004035272286273539f, 0.0004122326790820807f, 0.0004042254004161805f, 0.0013271215138956904f, 0.0004022294015157968f, 0.00041334546403959394f, 0.00103519216645509f, 0.0011803022352978587f, 0.0018181025516241789f, 0.0013995446497574449f, 0.0012068578507751226f, 0.0012130042305216193f, 0.0006532971747219563f, 0.0017677510622888803f, 0.0018439196282997727f, 0.0013359009753912687f, 0.00042384862899780273f, 0.0005125427851453424f, 0.0017284898785874248f, 0.0017989568877965212f, 0.0004964184481650591f, 0.0004193129134364426f, 0.00041465487447567284f, 0.0004935138858854771f, 0.0017815480241551995f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_head_head_2_Relu_output_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0902070477604866f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.058477964252233505f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020380117930471897f, 0.0019599541556090117f, 0.002109067514538765f, 0.002191092586144805f, 0.0025837451685220003f, 0.0018141827313229442f, 0.002056102268397808f, 0.0017579845152795315f, 0.002143032616004348f, 0.0019433973357081413f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2816, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1728, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch2_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_0_Flatten_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 672, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _head_head_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 10, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 10, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 82, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_Conv_output_0_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 8, 32), AI_STRIDE_INIT(4, 1, 1, 8, 64),
  1, &_features_features_0_Conv_output_0_output_array, &_features_features_0_Conv_output_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_Relu_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 8, 32), AI_STRIDE_INIT(4, 1, 1, 8, 64),
  1, &_features_features_1_Relu_output_0_output_array, &_features_features_1_Relu_output_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_features_features_2_Conv_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 4, 16), AI_STRIDE_INIT(4, 1, 1, 16, 64),
  1, &_features_features_2_Conv_output_0_output_array, &_features_features_2_Conv_output_0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 2816, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2816, 2816),
  1, &_features_features_2_Conv_output_0_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch1, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 2), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &_features_features_2_Conv_output_0_scratch1_array, &_features_features_2_Conv_output_0_scratch1_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_scratch2, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 2), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &_features_features_2_Conv_output_0_scratch2_array, &_features_features_2_Conv_output_0_scratch2_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_Conv_output_0_weights, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 16), AI_STRIDE_INIT(4, 1, 8, 128, 384),
  1, &_features_features_2_Conv_output_0_weights_array, &_features_features_2_Conv_output_0_weights_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_5_Conv_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 2, 8), AI_STRIDE_INIT(4, 1, 1, 32, 64),
  1, &_features_features_5_Conv_output_0_output_array, &_features_features_5_Conv_output_0_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_pad_before_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 6, 18), AI_STRIDE_INIT(4, 1, 1, 16, 96),
  1, &_features_features_5_Conv_output_0_pad_before_output_array, &_features_features_5_Conv_output_0_pad_before_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &_features_features_5_Conv_output_0_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch1, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 2), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &_features_features_5_Conv_output_0_scratch1_array, &_features_features_5_Conv_output_0_scratch1_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_scratch2, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 2), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &_features_features_5_Conv_output_0_scratch2_array, &_features_features_5_Conv_output_0_scratch2_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_Conv_output_0_weights, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &_features_features_5_Conv_output_0_weights_array, &_features_features_5_Conv_output_0_weights_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_0_Flatten_output_0_to_chlast_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 8, 32), AI_STRIDE_INIT(4, 1, 1, 2, 16),
  1, &_head_head_0_Flatten_output_0_to_chlast_output_array, &_head_head_0_Flatten_output_0_to_chlast_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_0_Flatten_output_0_to_chlast_output0, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &_head_head_0_Flatten_output_0_to_chlast_output_array, &_head_head_0_Flatten_output_0_to_chlast_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_head_head_1_Gemm_output_0_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_head_head_1_Gemm_output_0_output_array, &_head_head_1_Gemm_output_0_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 672, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1344, 1344),
  1, &_head_head_1_Gemm_output_0_scratch0_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_weights, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 512, 32, 1, 1), AI_STRIDE_INIT(4, 1, 512, 16384, 16384),
  1, &_head_head_1_Gemm_output_0_weights_array, &_head_head_1_Gemm_output_0_weights_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _head_head_2_Relu_output_0_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &_head_head_2_Relu_output_0_output_array, &_head_head_2_Relu_output_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &output_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &output_QuantizeLinear_Input_output_array, &output_QuantizeLinear_Input_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 82, 1, 1), AI_STRIDE_INIT(4, 2, 2, 164, 164),
  1, &output_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_weights, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 32, 10, 1, 1), AI_STRIDE_INIT(4, 1, 32, 320, 320),
  1, &output_QuantizeLinear_Input_weights_array, &output_QuantizeLinear_Input_weights_array_intq)



AI_STATIC_CONST ai_i8 _features_features_1_Relu_output_0_nl_params_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_1_Relu_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _features_features_1_Relu_output_0_nl_params_data, _features_features_1_Relu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_Relu_output_0_layer, 16,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu_integer,
  &_features_features_1_Relu_output_0_chain,
  NULL, &_features_features_1_Relu_output_0_layer, AI_STATIC, 
  .nl_params = &_features_features_1_Relu_output_0_nl_params, 
)


AI_STATIC_CONST ai_i8 _features_features_2_Conv_output_0_nl_params_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_2_Conv_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _features_features_2_Conv_output_0_nl_params_data, _features_features_2_Conv_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_Conv_output_0_weights, &_features_features_2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_Conv_output_0_scratch0, &_features_features_2_Conv_output_0_scratch1, &_features_features_2_Conv_output_0_scratch2)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_Conv_output_0_layer, 25,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &_features_features_2_Conv_output_0_chain,
  NULL, &_features_features_2_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = &_features_features_2_Conv_output_0_nl_params, 
  .nl_func = AI_HANDLE_PTR(nl_func_relu_array_integer), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _features_features_5_Conv_output_0_nl_params_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_5_Conv_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _features_features_5_Conv_output_0_nl_params_data, _features_features_5_Conv_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_5_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_5_Conv_output_0_weights, &_features_features_5_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_5_Conv_output_0_scratch0, &_features_features_5_Conv_output_0_scratch1, &_features_features_5_Conv_output_0_scratch2)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_5_Conv_output_0_layer, 34,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &_features_features_5_Conv_output_0_chain,
  NULL, &_features_features_5_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = &_features_features_5_Conv_output_0_nl_params, 
  .nl_func = AI_HANDLE_PTR(nl_func_relu_array_integer), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _head_head_0_Flatten_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_0_Flatten_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _head_head_0_Flatten_output_0_to_chlast_layer, 37,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_head_head_0_Flatten_output_0_to_chlast_chain,
  NULL, &_head_head_0_Flatten_output_0_to_chlast_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_0_Flatten_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_head_head_1_Gemm_output_0_weights, &_head_head_1_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_1_Gemm_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _head_head_1_Gemm_output_0_layer, 40,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_head_head_1_Gemm_output_0_chain,
  NULL, &_head_head_1_Gemm_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 _head_head_2_Relu_output_0_nl_params_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    _head_head_2_Relu_output_0_nl_params, AI_ARRAY_FORMAT_S8,
    _head_head_2_Relu_output_0_nl_params_data, _head_head_2_Relu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _head_head_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _head_head_2_Relu_output_0_layer, 43,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu_integer,
  &_head_head_2_Relu_output_0_chain,
  NULL, &_head_head_2_Relu_output_0_layer, AI_STATIC, 
  .nl_params = &_head_head_2_Relu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_head_head_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &output_QuantizeLinear_Input_weights, &output_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  output_QuantizeLinear_Input_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &output_QuantizeLinear_Input_chain,
  NULL, &output_QuantizeLinear_Input_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_relu_integer__features_features_1_Relu_output_0(_stai_network_context* net_ctx)
{
  _features_features_0_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2112);
  _features_features_0_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2112);
  _features_features_1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 64);
  _features_features_1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 64);
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, { _features_features_0_Conv_output_0_output.data->data});
  forward_relu_integer(&_features_features_1_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, { _features_features_1_Relu_output_0_output.data->data});
}
void forward_lite_conv2d_sssa8_ch_nl_pool__features_features_2_Conv_output_0(_stai_network_context* net_ctx)
{
  _features_features_1_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 64);
  _features_features_1_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 64);
  _features_features_2_Conv_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 104);
  _features_features_2_Conv_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 104);
  _features_features_2_Conv_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 1256);
  _features_features_2_Conv_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 1256);
  _features_features_2_Conv_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 2112);
  _features_features_2_Conv_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 2112);
  _features_features_2_Conv_output_0_scratch1_array.data = AI_PTR(net_ctx->_activations[0] + 4928);
  _features_features_2_Conv_output_0_scratch1_array.data_start = AI_PTR(net_ctx->_activations[0] + 4928);
  _features_features_2_Conv_output_0_scratch2_array.data = AI_PTR(net_ctx->_activations[0] + 5184);
  _features_features_2_Conv_output_0_scratch2_array.data_start = AI_PTR(net_ctx->_activations[0] + 5184);
  _features_features_2_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 5440);
  _features_features_2_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5440);
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, { _features_features_1_Relu_output_0_output.data->data});
  forward_conv2d_sssa8_ch_nl_pool(&_features_features_2_Conv_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, { _features_features_2_Conv_output_0_output.data->data});
}
void forward_lite_conv2d_deep_3x3_sssa8_ch_nl_pool__features_features_5_Conv_output_0(_stai_network_context* net_ctx)
{
  _features_features_5_Conv_output_0_pad_before_output_array.data = AI_PTR(net_ctx->_activations[0] + 64);
  _features_features_5_Conv_output_0_pad_before_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 64);
  _features_features_5_Conv_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 1320);
  _features_features_5_Conv_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 1320);
  _features_features_5_Conv_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 5928);
  _features_features_5_Conv_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 5928);
  _features_features_5_Conv_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 1792);
  _features_features_5_Conv_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 1792);
  _features_features_5_Conv_output_0_scratch1_array.data = AI_PTR(net_ctx->_activations[0] + 7936);
  _features_features_5_Conv_output_0_scratch1_array.data_start = AI_PTR(net_ctx->_activations[0] + 7936);
  _features_features_5_Conv_output_0_scratch2_array.data = AI_PTR(net_ctx->_activations[0] + 7936);
  _features_features_5_Conv_output_0_scratch2_array.data_start = AI_PTR(net_ctx->_activations[0] + 7936);
  _features_features_5_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _features_features_5_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, { _features_features_5_Conv_output_0_pad_before_output.data->data});
   forward_conv2d_deep_3x3_sssa8_ch_nl_pool(&_features_features_5_Conv_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, { _features_features_5_Conv_output_0_output.data->data});
}
void forward_lite_transpose__head_head_0_Flatten_output_0_to_chlast(_stai_network_context* net_ctx)
{
  _features_features_5_Conv_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _features_features_5_Conv_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _head_head_0_Flatten_output_0_to_chlast_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  _head_head_0_Flatten_output_0_to_chlast_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, { _features_features_5_Conv_output_0_output.data->data});
  forward_transpose(&_head_head_0_Flatten_output_0_to_chlast_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, { _head_head_0_Flatten_output_0_to_chlast_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch__head_head_1_Gemm_output_0(_stai_network_context* net_ctx)
{
  _head_head_0_Flatten_output_0_to_chlast_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  _head_head_0_Flatten_output_0_to_chlast_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  _head_head_1_Gemm_output_0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 6056);
  _head_head_1_Gemm_output_0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 6056);
  _head_head_1_Gemm_output_0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 22440);
  _head_head_1_Gemm_output_0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 22440);
  _head_head_1_Gemm_output_0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  _head_head_1_Gemm_output_0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  _head_head_1_Gemm_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _head_head_1_Gemm_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, { _head_head_0_Flatten_output_0_to_chlast_output0.data->data});
  forward_dense_integer_SSSA_ch(&_head_head_1_Gemm_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { _head_head_1_Gemm_output_0_output.data->data});
}
void forward_lite_relu_integer__head_head_2_Relu_output_0(_stai_network_context* net_ctx)
{
  _head_head_1_Gemm_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  _head_head_1_Gemm_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _head_head_2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 32);
  _head_head_2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 1, { _head_head_1_Gemm_output_0_output.data->data});
  forward_relu_integer(&_head_head_2_Relu_output_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { _head_head_2_Relu_output_0_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_output_QuantizeLinear_Input(_stai_network_context* net_ctx)
{
  _head_head_2_Relu_output_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 32);
  _head_head_2_Relu_output_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32);
  output_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 22568);
  output_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 22568);
  output_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 22888);
  output_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 22888);
  output_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 64);
  output_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 64);
  output_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  output_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, { _head_head_2_Relu_output_0_output.data->data});
  forward_dense_integer_SSSA_ch(&output_QuantizeLinear_Input_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, { output_QuantizeLinear_Input_output.data->data});
}

/*****************************************************************************/


static const ai_u16 _features_features_0_Conv_output_0_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 _features_features_0_Conv_output_0_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 _features_features_0_Conv_output_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 _features_features_0_Conv_output_0_t_out_0_shape_ch_const_u16 = 8;
static const ai_u16 _features_features_0_Conv_output_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 _features_features_0_Conv_output_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 _features_features_0_Conv_output_0_l_stride_1_const_u16 = 1;
static const ai_u16 _features_features_0_Conv_output_0_l_stride_0_const_u16 = 1;
static const ai_i32 _features_features_0_Conv_output_0_l_pad_W_0_const_s32 = 1;
static const ai_i32 _features_features_0_Conv_output_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 _features_features_0_Conv_output_0_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 _features_features_0_Conv_output_0_t_out_0_fmt_zero_const_s8 = 0;
static const ai_float _features_features_0_Conv_output_0_t_in_0_fmt_scale_const_f32 = 0.007874015718698502f;
static const ai_float _features_features_0_Conv_output_0_t_out_0_fmt_scale_const_f32 = 0.013158309273421764f;
static const ai_float _features_features_0_Conv_output_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0030700741335749626f, 0.0030698811169713736f, 0.002619256032630801f, 0.00406322255730629f, 0.003483925713226199f, 0.0038459193892776966f, 0.0033700400963425636f, 0.0027155503630638123f);
static const ai_layer_format_type _features_features_0_Conv_output_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 _features_features_0_Conv_output_0_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 _features_features_0_Conv_output_0_t_out_0_shape_h_const_u16 = 32;



static const ai_i8 _features_features_5_Conv_output_0_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(0);
static const ai_i16 _features_features_5_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 _features_features_5_Conv_output_0_pad_before_t_in_0_shape_h_const_u32 = 16;





STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN _features_features_0_Conv_output_0 */
  {
      const ai_i8* _features_features_0_Conv_output_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* _features_features_0_Conv_output_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* _features_features_0_Conv_output_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 72);
    ai_i8* _features_features_0_Conv_output_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 2112);
    ai_i16* _features_features_0_Conv_output_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1820);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) _features_features_0_Conv_output_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(_features_features_0_Conv_output_0_t_in_0_ptr_const_s8, _features_features_0_Conv_output_0_t_in_0_shape_w_const_u16, _features_features_0_Conv_output_0_t_in_0_shape_h_const_u16, _features_features_0_Conv_output_0_t_in_0_shape_ch_const_u16, _features_features_0_Conv_output_0_t_weight_0_ptr_const_s8, _features_features_0_Conv_output_0_t_out_0_shape_ch_const_u16, _features_features_0_Conv_output_0_t_weight_0_shape_w_const_u16, _features_features_0_Conv_output_0_t_weight_0_shape_h_const_u16, _features_features_0_Conv_output_0_l_stride_1_const_u16, _features_features_0_Conv_output_0_l_stride_0_const_u16, _features_features_0_Conv_output_0_l_pad_W_0_const_s32, _features_features_0_Conv_output_0_l_pad_H_0_const_s32, _features_features_0_Conv_output_0_t_weight_1_ptr_const_s32, _features_features_0_Conv_output_0_t_in_0_fmt_zero_const_s8, _features_features_0_Conv_output_0_t_out_0_fmt_zero_const_s8, _features_features_0_Conv_output_0_t_in_0_fmt_scale_const_f32, _features_features_0_Conv_output_0_t_out_0_fmt_scale_const_f32, _features_features_0_Conv_output_0_t_weight_0_fmt_scale_const_f32, _features_features_0_Conv_output_0_l_out_ch_format_const_layer_format_type, _features_features_0_Conv_output_0_t_out_0_ptr_s8, _features_features_0_Conv_output_0_t_out_0_shape_w_const_u16, _features_features_0_Conv_output_0_t_out_0_shape_h_const_u16, 1, 292, _features_features_0_Conv_output_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) _features_features_0_Conv_output_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END _features_features_0_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _features_features_1_Relu_output_0 */
  {
    
  forward_lite_relu_integer__features_features_1_Relu_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _features_features_1_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _features_features_2_Conv_output_0 */
  {
    
  forward_lite_conv2d_sssa8_ch_nl_pool__features_features_2_Conv_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _features_features_2_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _features_features_5_Conv_output_0_pad_before */
  {
      const ai_ptr _features_features_5_Conv_output_0_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 5440);
    ai_ptr _features_features_5_Conv_output_0_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, {(stai_ptr) _features_features_5_Conv_output_0_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(_features_features_5_Conv_output_0_pad_before_t_in_0_ptr_const_ptr, _features_features_5_Conv_output_0_pad_before_t_out_0_ptr_ptr, (ai_handle)(_features_features_5_Conv_output_0_pad_before_v_pad_constant_value_const_s8), _features_features_5_Conv_output_0_pad_before_t_in_0_fmt_bitsize_const_s16, _features_features_5_Conv_output_0_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(64), (ai_i32)(96), (ai_i32)(96), (ai_i32)(16), (ai_i32)(16));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, {(stai_ptr) _features_features_5_Conv_output_0_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END _features_features_5_Conv_output_0_pad_before */
  /* LITE_KERNEL_SECTION BEGIN _features_features_5_Conv_output_0 */
  {
    
  forward_lite_conv2d_deep_3x3_sssa8_ch_nl_pool__features_features_5_Conv_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _features_features_5_Conv_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _head_head_0_Flatten_output_0_to_chlast */
  {
    
  forward_lite_transpose__head_head_0_Flatten_output_0_to_chlast(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _head_head_0_Flatten_output_0_to_chlast */
  /* LITE_KERNEL_SECTION BEGIN _head_head_1_Gemm_output_0 */
  {
    
  forward_lite_dense_integer_SSSA_ch__head_head_1_Gemm_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _head_head_1_Gemm_output_0 */
  /* LITE_KERNEL_SECTION BEGIN _head_head_2_Relu_output_0 */
  {
    
  forward_lite_relu_integer__head_head_2_Relu_output_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _head_head_2_Relu_output_0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input */
  {
    
  forward_lite_dense_integer_SSSA_ch_output_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END output_QuantizeLinear_Input */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 1564;

  net_ctx->_outputs[0] = activations[0] + 0;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

