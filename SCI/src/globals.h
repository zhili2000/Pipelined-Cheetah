/*
Authors: Nishant Kumar, Deevashwer Rathee
Modified by Wen-jie Lu
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GLOBALS_H___
#define GLOBALS_H___

#include "NonLinear/argmax.h"
#include "NonLinear/maxpool.h"
#include "NonLinear/relu-interface.h"
#include "defines.h"
#include "defines_uniform.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include "OT/kkot.h"
#ifdef SCI_OT
#include "BuildingBlocks/aux-protocols.h"
#include "BuildingBlocks/truncation.h"
#include "LinearOT/linear-ot.h"
#include "LinearOT/linear-uniform.h"
#include "Math/math-functions.h"
#endif
// Additional Headers for Athos
#ifdef SCI_HE
#include "LinearHE/elemwise-prod-field.h"
#include "LinearHE/fc-field.h"
#include "LinearHE/conv-field.h"
#endif

#if USE_CHEETAH
#include "cheetah/cheetah-api.h"
#endif

// #define MULTI_THREADING

#define MAX_THREADS 4

// Maximum batch size allowed
#define MAX_BATCH 10

extern sci::NetIO *io[MAX_BATCH];
extern sci::OTPack<sci::NetIO> *otpack[MAX_BATCH];

#ifdef SCI_OT
extern LinearOT *mult[MAX_BATCH];
extern AuxProtocols *aux[MAX_BATCH];
extern Truncation *truncation[MAX_BATCH];
extern XTProtocol *xt[MAX_BATCH];
extern MathFunctions *math[MAX_BATCH];
#endif
extern ArgMaxProtocol<sci::NetIO, intType> *argmax[MAX_BATCH];
extern ReLUProtocol<sci::NetIO, intType> *relu[MAX_BATCH];
extern MaxPoolProtocol<sci::NetIO, intType> *maxpool[MAX_BATCH];
// Additional classes for Athos

#ifdef SCI_OT
extern MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniform[MAX_BATCH];
#elif defined(SCI_HE)
extern FCField *he_fc[MAX_BATCH];
extern ElemWiseProdField *he_prod[MAX_BATCH];
#endif

#if USE_CHEETAH
extern gemini::CheetahLinear *cheetah_linear[MAX_BATCH];
extern bool kIsSharedInput[MAX_BATCH];
#elif defined(SCI_HE)
extern ConvField *he_conv[MAX_BATCH];
#endif

extern sci::IKNP<sci::NetIO> *iknpOT[MAX_BATCH];
extern sci::IKNP<sci::NetIO> *iknpOTRoleReversed[MAX_BATCH];
extern sci::KKOT<sci::NetIO> *kkot[MAX_BATCH];
extern sci::PRG128 *prg128Instance[MAX_BATCH];

extern sci::NetIO *ioArr[MAX_THREADS * MAX_BATCH];
extern sci::OTPack<sci::NetIO> *otpackArr[MAX_THREADS * MAX_BATCH];
#ifdef SCI_OT
extern LinearOT *multArr[MAX_THREADS * MAX_BATCH];
extern AuxProtocols *auxArr[MAX_THREADS * MAX_BATCH];
extern Truncation *truncationArr[MAX_THREADS * MAX_BATCH];
extern XTProtocol *xtArr[MAX_THREADS * MAX_BATCH];
extern MathFunctions *mathArr[MAX_THREADS * MAX_BATCH];
#endif
extern ReLUProtocol<sci::NetIO, intType> *reluArr[MAX_THREADS * MAX_BATCH];
extern MaxPoolProtocol<sci::NetIO, intType> *maxpoolArr[MAX_THREADS * MAX_BATCH];
// Additional classes for Athos
#ifdef SCI_OT
extern MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>>
    *multUniformArr[MAX_THREADS * MAX_BATCH];
#endif
extern sci::IKNP<sci::NetIO> *otInstanceArr[MAX_THREADS * MAX_BATCH];
extern sci::KKOT<sci::NetIO> *kkotInstanceArr[MAX_THREADS * MAX_BATCH];
extern sci::PRG128 *prgInstanceArr[MAX_THREADS * MAX_BATCH];

extern std::chrono::time_point<std::chrono::high_resolution_clock> start_time[MAX_BATCH];
extern uint64_t comm_threads[MAX_THREADS * MAX_BATCH];
extern uint64_t num_rounds[MAX_BATCH];

#ifdef LOG_LAYERWISE
extern uint64_t ConvTimeInMilliSec[MAX_BATCH];
extern uint64_t MatAddTimeInMilliSec[MAX_BATCH];
extern uint64_t BatchNormInMilliSec[MAX_BATCH];
extern uint64_t TruncationTimeInMilliSec[MAX_BATCH];
extern uint64_t ReluTimeInMilliSec[MAX_BATCH];
extern uint64_t MaxpoolTimeInMilliSec[MAX_BATCH];
extern uint64_t AvgpoolTimeInMilliSec[MAX_BATCH];
extern uint64_t MatMulTimeInMilliSec[MAX_BATCH];
extern uint64_t MatAddBroadCastTimeInMilliSec[MAX_BATCH];
extern uint64_t MulCirTimeInMilliSec[MAX_BATCH];
extern uint64_t ScalarMulTimeInMilliSec[MAX_BATCH];
extern uint64_t SigmoidTimeInMilliSec[MAX_BATCH];
extern uint64_t TanhTimeInMilliSec[MAX_BATCH];
extern uint64_t SqrtTimeInMilliSec[MAX_BATCH];
extern uint64_t NormaliseL2TimeInMilliSec[MAX_BATCH];
extern uint64_t ArgMaxTimeInMilliSec[MAX_BATCH];

extern uint64_t ConvCommSent[MAX_BATCH];
extern uint64_t MatAddCommSent[MAX_BATCH];
extern uint64_t BatchNormCommSent[MAX_BATCH];
extern uint64_t TruncationCommSent[MAX_BATCH];
extern uint64_t ReluCommSent[MAX_BATCH];
extern uint64_t MaxpoolCommSent[MAX_BATCH];
extern uint64_t AvgpoolCommSent[MAX_BATCH];
extern uint64_t MatMulCommSent[MAX_BATCH];
extern uint64_t MatAddBroadCastCommSent[MAX_BATCH];
extern uint64_t MulCirCommSent[MAX_BATCH];
extern uint64_t ScalarMulCommSent[MAX_BATCH];
extern uint64_t SigmoidCommSent[MAX_BATCH];
extern uint64_t TanhCommSent[MAX_BATCH];
extern uint64_t SqrtCommSent[MAX_BATCH];
extern uint64_t NormaliseL2CommSent[MAX_BATCH];
extern uint64_t ArgMaxCommSent[MAX_BATCH];
#endif

#endif // GLOBALS_H__
