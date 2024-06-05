/*
Authors: Nishant Kumar, Deevashwer Rathee
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

#include "globals.h"

sci::NetIO *io[MAX_BATCH];
sci::OTPack<sci::NetIO> *otpack[MAX_BATCH];

#ifdef SCI_OT
LinearOT *mult[MAX_BATCH];
AuxProtocols *aux[MAX_BATCH];
Truncation *truncation[MAX_BATCH];
XTProtocol *xt[MAX_BATCH];
MathFunctions *math[MAX_BATCH];
#endif
ArgMaxProtocol<sci::NetIO, intType> *argmax[MAX_BATCH];
ReLUProtocol<sci::NetIO, intType> *relu[MAX_BATCH];
MaxPoolProtocol<sci::NetIO, intType> *maxpool[MAX_BATCH];
// Additional classes for Athos
#ifdef SCI_OT
MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniform[MAX_BATCH];
#endif

#ifdef SCI_HE
FCField *he_fc[MAX_BATCH];
ElemWiseProdField *he_prod[MAX_BATCH];
#endif

#if USE_CHEETAH
gemini::CheetahLinear *cheetah_linear[MAX_BATCH];
bool kIsSharedInput;
#elif defined(SCI_HE)
ConvField *he_conv[MAX_BATCH];
#endif

sci::IKNP<sci::NetIO> *iknpOT[MAX_BATCH];
sci::IKNP<sci::NetIO> *iknpOTRoleReversed[MAX_BATCH];
sci::KKOT<sci::NetIO> *kkot[MAX_BATCH];
sci::PRG128 *prg128Instance[MAX_BATCH];

sci::NetIO *ioArr[MAX_THREADS * MAX_BATCH];
sci::OTPack<sci::NetIO> *otpackArr[MAX_THREADS * MAX_BATCH];
#ifdef SCI_OT
LinearOT *multArr[MAX_THREADS * MAX_BATCH];
AuxProtocols *auxArr[MAX_THREADS * MAX_BATCH];
Truncation *truncationArr[MAX_THREADS * MAX_BATCH];
XTProtocol *xtArr[MAX_THREADS * MAX_BATCH];
MathFunctions *mathArr[MAX_THREADS * MAX_BATCH];
#endif
ReLUProtocol<sci::NetIO, intType> *reluArr[MAX_THREADS * MAX_BATCH];
MaxPoolProtocol<sci::NetIO, intType> *maxpoolArr[MAX_THREADS * MAX_BATCH];
// Additional classes for Athos
#ifdef SCI_OT
MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniformArr[MAX_THREADS * MAX_BATCH];
#endif
sci::IKNP<sci::NetIO> *otInstanceArr[MAX_THREADS * MAX_BATCH];
sci::KKOT<sci::NetIO> *kkotInstanceArr[MAX_THREADS * MAX_BATCH];
sci::PRG128 *prgInstanceArr[MAX_THREADS * MAX_BATCH];

std::chrono::time_point<std::chrono::high_resolution_clock> start_time[MAX_BATCH];
uint64_t comm_threads[MAX_THREADS * MAX_BATCH];
uint64_t num_rounds[MAX_BATCH];

#ifdef LOG_LAYERWISE
uint64_t ConvTimeInMilliSec[MAX_BATCH] = {0};
uint64_t MatAddTimeInMilliSec[MAX_BATCH] = {0};
uint64_t BatchNormInMilliSec[MAX_BATCH] = {0};
uint64_t TruncationTimeInMilliSec[MAX_BATCH] = {0};
uint64_t ReluTimeInMilliSec[MAX_BATCH] = {0};
uint64_t MaxpoolTimeInMilliSec[MAX_BATCH] = {0};
uint64_t AvgpoolTimeInMilliSec[MAX_BATCH] = {0};
uint64_t MatMulTimeInMilliSec[MAX_BATCH] = {0};
uint64_t MatAddBroadCastTimeInMilliSec[MAX_BATCH] = {0};
uint64_t MulCirTimeInMilliSec[MAX_BATCH] = {0};
uint64_t ScalarMulTimeInMilliSec[MAX_BATCH] = {0};
uint64_t SigmoidTimeInMilliSec[MAX_BATCH] = {0};
uint64_t TanhTimeInMilliSec[MAX_BATCH] = {0};
uint64_t SqrtTimeInMilliSec[MAX_BATCH] = {0};
uint64_t NormaliseL2TimeInMilliSec[MAX_BATCH] = {0};
uint64_t ArgMaxTimeInMilliSec[MAX_BATCH] = {0};

uint64_t ConvCommSent[MAX_BATCH] = {0};
uint64_t MatAddCommSent[MAX_BATCH] = {0};
uint64_t BatchNormCommSent[MAX_BATCH] = {0};
uint64_t TruncationCommSent[MAX_BATCH] = {0};
uint64_t ReluCommSent[MAX_BATCH] = {0};
uint64_t MaxpoolCommSent[MAX_BATCH] = {0};
uint64_t AvgpoolCommSent[MAX_BATCH] = {0};
uint64_t MatMulCommSent[MAX_BATCH] = {0};
uint64_t MatAddBroadCastCommSent[MAX_BATCH] = {0};
uint64_t MulCirCommSent[MAX_BATCH] = {0};
uint64_t ScalarMulCommSent[MAX_BATCH] = {0};
uint64_t SigmoidCommSent[MAX_BATCH] = {0};
uint64_t TanhCommSent[MAX_BATCH] = {0};
uint64_t SqrtCommSent[MAX_BATCH] = {0};
uint64_t NormaliseL2CommSent[MAX_BATCH] = {0};
uint64_t ArgMaxCommSent[MAX_BATCH] = {0};
#endif
