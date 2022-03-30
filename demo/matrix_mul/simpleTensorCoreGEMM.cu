/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <pthread.h>
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
/*#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384
#define MATRIX_M 10240
#define MATRIX_N 10240
#define MATRIX_K 10240
*/
/*#define MATRIX_M 2048
#define MATRIX_N 2048
#define MATRIX_K 2048*/



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n, float scale) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      if(scale > 32768.0)
          out[idx] = (half)(in[idx]/scale);
      else
      out[idx] = in[idx];
   }
}

__global__ void scaleRandom (float *out, float *in, int n, float scale) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
          out[idx] = (int)(in[idx]*scale);
   }
}

void* cublasCreateThread(void *x)
{
//   struct timeval time_start, time_end, total_start, total_end;
//   int init_time;
   cublasHandle_t* cublasHandle = (cublasHandle_t *)x;
//    gettimeofday(&time_start, NULL);
   cublasErrCheck(cublasCreate(cublasHandle));
//   cublasErrCheck(cublasSetMathMode(*cublasHandle, CUBLAS_TENSOR_OP_MATH));
//	fprintf(stderr,"cublasCreate\n");
//  	gettimeofday(&time_end, NULL);
//        init_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
//	fprintf(stderr,"cublasCreate  %d\n",init_time);
  return NULL;
}
void* cublasCreateDefaultThread(void *x)
{
//   struct timeval time_start, time_end, total_start, total_end;
//   int init_time;
   cublasHandle_t* cublasHandle = (cublasHandle_t *)x;
//    gettimeofday(&time_start, NULL);
   cublasErrCheck(cublasCreate(cublasHandle));
//   cublasErrCheck(cublasSetMathMode(*cublasHandle, CUBLAS_DEFAULT_MATH));
//  	gettimeofday(&time_end, NULL);
//        init_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
//	fprintf(stderr,"cublasCreate (2) %d\n",init_time);
  return NULL;
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   float scale;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;
   float *c_sgemm;
   float *c_cublas_gemmEx;

   float *c_host_cublas;
   float *c_host_cublasCublasGemmEx;
   float *c_host_wmma;
   float *c_host_sgemm;

   float alpha = 1.0f;
   float beta = 0.0f;

int MATRIX_M=10240;
int MATRIX_N=10240;
int MATRIX_K=10240;
   
   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   cublasHandle_t cublasHandle_default;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaEvent_t startcublasEX;
   cudaEvent_t stopcublasEX;

   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaEvent_t startcublasCublasGemmEx;
   cudaEvent_t stopcublasCublasGemmEx;
    struct timeval time_start, time_end, total_start, total_end;
   int init_time;
    gettimeofday(&time_start, NULL);
    cudaFree(0);
   #ifdef THREAD
  pthread_t thread;
  pthread_t thread1;
  pthread_create(&thread, NULL, cublasCreateThread, &cublasHandle);
  pthread_create(&thread1, NULL, cublasCreateDefaultThread, &cublasHandle_default);
   #endif  
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));

   cudaErrCheck(cudaEventCreate(&startcublasEX));
   cudaErrCheck(cudaEventCreate(&stopcublasEX));

   cudaErrCheck(cudaEventCreate(&startcublasCublasGemmEx));
   cudaErrCheck(cudaEventCreate(&stopcublasCublasGemmEx));
   
   #ifndef THREAD
    gettimeofday(&total_start, NULL);
   cublasErrCheck(cublasCreate(&cublasHandle));
   cublasErrCheck(cublasCreate(&cublasHandle_default));
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));
  	gettimeofday(&total_end, NULL);
        init_time = ((total_end.tv_sec * 1000000 + total_end.tv_usec) - (total_start.tv_sec * 1000000 + total_start.tv_usec));
	fprintf(stderr,"cublasCreate(2)  %d\n",init_time);
   #endif
   MATRIX_M = atoi(argv[1]);
   MATRIX_K = MATRIX_M;
   MATRIX_N = MATRIX_M;
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_sgemm, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_cublasCublasGemmEx = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_sgemm = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));
   scale = atof(argv[2]);
   if(scale == 0.0)
       scale=3.4028234664e+38;
   printf("Scale: %f\n",scale);
   scaleRandom <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp32, a_fp32, MATRIX_M * MATRIX_K, scale);
   scaleRandom <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (b_fp32, b_fp32, MATRIX_N * MATRIX_K, scale);
   
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K, scale);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N, scale);

   curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   curandErrCheck(curandDestroyGenerator(gen));
   
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_cublas_gemmEx, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
#ifdef THREAD
  pthread_join(thread, NULL); 
  pthread_join(thread1, NULL); 
//   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   cublasErrCheck(cublasSetMathMode(cublasHandle_default, CUBLAS_DEFAULT_MATH));
#endif
  	gettimeofday(&time_end, NULL);
        init_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"Before GEMM %d\n",init_time);


   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   printf("Running with sgemm...\n");
   cudaErrCheck(cudaEventRecord(startcublas));
   cublasSgemm(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp32, MATRIX_M, b_fp32, MATRIX_N, &beta, c_sgemm, MATRIX_K);
   cudaErrCheck(cudaEventRecord(stopcublas));
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);


   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   cudaErrCheck(cudaEventRecord(stopWMMA));


   // Now using cuBLAS but not tensor
   printf("Running with cuBLAS (GemmEX)...\n");
   cudaErrCheck(cudaEventRecord(startcublasCublasGemmEx));
   cublasErrCheck(cublasGemmEx(cublasHandle_default, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas_gemmEx, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT));
   cudaErrCheck(cudaEventRecord(stopcublasCublasGemmEx));

//   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
   
   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   cudaErrCheck(cudaEventRecord(startcublasEX));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
   cudaErrCheck(cudaEventRecord(stopcublasEX));

   // Error checking
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_cublasCublasGemmEx, c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c_host_sgemm, c_sgemm, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

   printf("\nChecking results with cublas (cublasGemmEx)...\n");
   int errors_default = 0;
   double error_rate =0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_cublasCublasGemmEx[i];
      float v2 = c_host_sgemm[i];
      if(scale > 32768)
          v1 = v1*scale*scale;
      //    MATRIX_M * MATRIX_N;
      error_rate += (abs(v1 - v2)/v1);
      if (abs(v1 - v2) > 1e-5) {
//      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
         errors_default++;
         if (errors_default < 3) printf("%f %f\n", v1, v2);
      }
   }
   if (errors_default > 0) {
      printf("GemmEX does not agree with cuBLAS default! %d errors -- error rate %lf!\n", errors_default/MATRIX_M * MATRIX_N,error_rate/errors_default);
   }
   
   printf("\nChecking results with tensor cores...\n");
   // 0.01% relative tolerance. 1e-5 absolute tolerance.
   int errors = 0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_wmma[i];
      float v2 = c_host_sgemm[i];
//      float v2 = c_host_cublasCublasGemmEx[i];
      error_rate += (abs(v1 - v2)/v1);
//      float v2 = c_host_cublasCublasGemmEx[i];
      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
         errors++;
         if (errors < 3) printf("%f %f\n", v1, v2);
      }
   }

   if (errors > 0) {
      printf("WMMA does not agree with cuBLAS! %d errors!-- error rate %lf!\n", errors_default,error_rate/errors_default);
   }
   errors = 0;
   error_rate =0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
//      float v1 = c_host_sgemm[i];
      float v1 = c_host_cublas[i];
      float v2 = c_host_sgemm[i];
     if(scale > 32768)
          v1 = v1*scale*scale;
 //      float v2 = c_host_cublasCublasGemmEx[i];
      error_rate += (abs(v1 - v2)/v1);
//      float v2 = c_host_cublasCublasGemmEx[i];
      if (abs(v1 - v2) > 1e-5) {
//      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-3) {
         errors++;
         if (errors < 3) printf("%f %f\n", v1, v2);
      }
   }

   if (errors > 0) {
      printf("cuBLAS TCU not agree with cuBLAS! %.10lf errors!-- error rate %.10lf!\n", (float)errors/(MATRIX_M * MATRIX_N),error_rate/errors);
   }

//   else {
//   {
//      printf("Results verified: cublas and WMMA agree.\n\n");
      float wmmaTime;
      float cublasTime;
      cudaErrCheck(cudaEventSynchronize(stopWMMA));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("wmma took %fms\n", wmmaTime);
      printf("cublas took %fms\n", cublasTime);
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasCublasGemmEx, stopcublasCublasGemmEx));
      printf("cublas cublasGemmEx took %fms\n", cublasTime);
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasEX, stopcublasEX));
      printf("cublas tensor cores took %fms\n", cublasTime);

      printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
//   }
   printf("Running with cuBLASCdot...\n");
   cudaEvent_t startDot;
   cudaEvent_t stopDot;
   cudaErrCheck(cudaEventCreate(&startDot));
   cudaErrCheck(cudaEventCreate(&stopDot));

   cublasSetPointerMode(cublasHandle_default,CUBLAS_POINTER_MODE_DEVICE); // set here!!!
   cudaErrCheck(cudaEventRecord(startDot));
   for(int i = 0; i< MATRIX_M; i++)
   {
       cublasErrCheck(cublasSdot(cublasHandle_default,MATRIX_N,&a_fp32[i*MATRIX_N],1,&b_fp32[i*MATRIX_N],1,&c_cublas_gemmEx[i*MATRIX_N]));
   }
   cudaErrCheck(cudaEventRecord(stopDot));
      cudaErrCheck(cudaEventSynchronize(stopDot));

   cudaErrCheck(cudaEventElapsedTime(&cublasTime, startDot, stopDot));
   printf("cublas dot product took %fms\n", cublasTime);
   
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   cudaErrCheck(cudaFree(c_wmma));
   
   free(c_host_cublas);
   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


