// Strassen one level, fused to 3 kernel launches:
//   1) pack: build 7 A-panels + 7 B-panels (one kernel over 7*me elements each)
//   2) one cublasGemmStridedBatchedEx with batchCount=7  <- the whole 7 multiplications
//   3) combine: 4 quadrants from the 7 M panels (one kernel)
// Purpose: measure the *achievable* 8/7 speedup ceiling without launch overhead dominating.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cublasStatus_t s_ = (cublasStatus_t)(x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  printf("ERR %s line %d code %d\n", #x, __LINE__, (int)s_); exit(1);} } while (0)
#define CKU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("ERR %s line %d : %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

// ---- pack kernel: 7 panels, each me = m*m ----
// A operand list (index -> expression), 4 blocks a11,a12,a21,a22
__device__ __forceinline__ __half pickA(int b, __half a11, __half a12, __half a21, __half a22) {
  switch (b) {
    case 0: return __hadd(a11, a22);
    case 1: return __hadd(a21, a22);
    case 2: return a11;
    case 3: return a22;
    case 4: return __hadd(a11, a12);
    case 5: return __hsub(a21, a11);
    default:return __hsub(a12, a22);
  }
}
__device__ __forceinline__ __half pickB(int b, __half b11, __half b12, __half b21, __half b22) {
  switch (b) {
    case 0: return __hadd(b11, b22);
    case 1: return b11;
    case 2: return __hsub(b12, b22);
    case 3: return __hsub(b21, b11);
    case 4: return b22;
    case 5: return __hadd(b11, b12);
    default:return __hadd(b21, b22);
  }
}
// each block is m x m, source ld = n, block (bi,bj) starts at bi*m + bj*m*ld
__global__ void pack7(const __half* __restrict__ A, __half* __restrict__ AP,
                      const __half* __restrict__ B, __half* __restrict__ BP,
                      int m, int n) {
  int me = m * m;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)7 * me) return;
  int b = (int)(i / me), k = (int)(i % me);
  int c = k / m, r = k % m;
  int o00 = r + c * n, o01 = r + (c + m) * n, o10 = (r + m) + c * n, o11 = (r + m) + (c + m) * n;
  AP[i] = pickA(b, A[o00], A[o01], A[o10], A[o11]);
  BP[i] = pickB(b, B[o00], B[o01], B[o10], B[o11]);
}

// combine: C11=M0+M3-M4+M6 ; C12=M2+M4 ; C21=M1+M3 ; C22=M0+M2-M1+M5
__global__ void combine7(const __half* __restrict__ MP, __half* __restrict__ C, int m, int n) {
  int me = m * m;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)4 * me) return;
  int q = (int)(i / me), k = (int)(i % me);
  int c = k / m, r = k % m;
  const __half* M0 = MP; const __half* M1 = MP + me; const __half* M2 = MP + 2 * me;
  const __half* M3 = MP + 3 * me; const __half* M4 = MP + 4 * me;
  const __half* M5 = MP + 5 * me; const __half* M6 = MP + 6 * me;
  __half v;
  if (q == 0) v = __hadd(__hadd(__hsub(__hadd(M0[k], M3[k]), M4[k]), M6[k]), __float2half(0.f));
  else if (q == 1) v = __hadd(M2[k], M4[k]);
  else if (q == 2) v = __hadd(M1[k], M3[k]);
  else v = __hadd(__hsub(__hadd(M0[k], M2[k]), M1[k]), M5[k]);
  int bi = q / 2, bj = q % 2;              // quadrant order: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
  C[(long)(r + bi * m) + (long)(c + bj * m) * n] = v;
}

static void gemm(cublasHandle_t h, int n, const __half* A, const __half* B, __half* C, int reps) {
  float al = 1.f, be = 0.f;
  for (int i = 0; i < reps; i++)
    CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, A, CUDA_R_16F, n,
                    B, CUDA_R_16F, n, &be, C, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CKU(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2048;
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  if (n % 2) { printf("n must be even\n"); return 1; }
  int m = n / 2, me = m * m;

  __half *A, *B, *C1, *C2, *AP, *BP, *MP;
  CKU(cudaMalloc(&A, (size_t)n * n * 2));  CKU(cudaMalloc(&B, (size_t)n * n * 2));
  CKU(cudaMalloc(&C1, (size_t)n * n * 2)); CKU(cudaMalloc(&C2, (size_t)n * n * 2));
  CKU(cudaMalloc(&AP, (size_t)7 * me * 2)); CKU(cudaMalloc(&BP, (size_t)7 * me * 2));
  CKU(cudaMalloc(&MP, (size_t)7 * me * 2));

  // deterministic pseudo-random fill on host
  __half *hA = (__half*)malloc((size_t)n * n * 2), *hB = (__half*)malloc((size_t)n * n * 2);
  srand(1234);
  for (long i = 0; i < (long)n * n; i++) {
    hA[i] = __float2half((float)((rand() % 2001) - 1000) / 1000.f);
    hB[i] = __float2half((float)((rand() % 2001) - 1000) / 1000.f);
  }
  CKU(cudaMemcpy(A, hA, (size_t)n * n * 2, cudaMemcpyHostToDevice));
  CKU(cudaMemcpy(B, hB, (size_t)n * n * 2, cudaMemcpyHostToDevice));

  cublasHandle_t h; CK(cublasCreate(&h));
  int thr = 256, blocks7 = (7 * me + thr - 1) / thr, blocks4 = (4 * me + thr - 1) / thr;
  float al = 1.f, be = 0.f;

  auto run_strassen = [&]() {
    pack7<<<blocks7, thr>>>(A, AP, B, BP, m, n);
    CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &al,
        AP, CUDA_R_16F, m, (long long)me, BP, CUDA_R_16F, m, (long long)me,
        &be, MP, CUDA_R_16F, m, (long long)me, 7, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    combine7<<<blocks4, thr>>>(MP, C2, m, n);
  };

  // correctness
  gemm(h, n, A, B, C1, 1);
  run_strassen(); CKU(cudaDeviceSynchronize());

  __half *hC1 = (__half*)malloc((size_t)n * n * 2), *hC2 = (__half*)malloc((size_t)n * n * 2);
  CKU(cudaMemcpy(hC1, C1, (size_t)n * n * 2, cudaMemcpyDeviceToHost));
  CKU(cudaMemcpy(hC2, C2, (size_t)n * n * 2, cudaMemcpyDeviceToHost));
  double mx = 0, dmax = 0; int dr = 0, dc = 0;
  for (int c = 0; c < n; c++) for (int r = 0; r < n; r++) {
    long i = (long)r + (long)c * n;
    double a = __half2float(hC1[i]), b = __half2float(hC2[i]);
    if (fabs(a) > mx) mx = fabs(a);
    if (fabs(a - b) > dmax) { dmax = fabs(a - b); dr = r; dc = c; }
  }
  printf("n=%d  |C|max=%.3f  max|fused-cublas|=%.4f at (%d,%d)\n", n, mx, dmax, dr, dc);
  // per-quadrant diff, plus M-panel check against CPU for small n
  double qmax[4] = {0, 0, 0, 0};
  for (int c = 0; c < n; c++) for (int r = 0; r < n; r++) {
    long i = (long)r + (long)c * n;
    double d = fabs(__half2float(hC1[i]) - __half2float(hC2[i]));
    int q = (r >= m ? 2 : 0) + (c >= m ? 1 : 0);
    if (d > qmax[q]) qmax[q] = d;
  }
  printf("  quad diffs: %s=%.4f %s=%.4f %s=%.4f %s=%.4f\n", "C11", qmax[0], "C12", qmax[1], "C21", qmax[2], "C22", qmax[3]);
  if (n <= 512) {   // CPU verify of the 7 panels
    // (a) verify the pack kernel itself
    pack7<<<blocks7, thr>>>(A, AP, B, BP, m, n); CKU(cudaDeviceSynchronize());
    __half* hAP = (__half*)malloc((size_t)7 * me * 2);
    __half* hBP = (__half*)malloc((size_t)7 * me * 2);
    CKU(cudaMemcpy(hAP, AP, (size_t)7 * me * 2, cudaMemcpyDeviceToHost));
    CKU(cudaMemcpy(hBP, BP, (size_t)7 * me * 2, cudaMemcpyDeviceToHost));
    double pa[7] = {0,0,0,0,0,0,0}, pb[7] = {0,0,0,0,0,0,0};
    for (int b = 0; b < 7; b++) for (int c = 0; c < m; c++) for (int r = 0; r < m; r++) {
      long o00 = (long)r + (long)c * n, o01 = (long)r + (long)(c + m) * n;
      long o10 = (long)(r + m) + (long)c * n, o11 = (long)(r + m) + (long)(c + m) * n;
      double a11 = __half2float(hA[o00]), a12 = __half2float(hA[o01]);
      double a21 = __half2float(hA[o10]), a22 = __half2float(hA[o11]);
      double b11 = __half2float(hB[o00]), b21 = __half2float(hB[o10]);
      double b12 = __half2float(hB[o01]), b22 = __half2float(hB[o11]);
      double av, bv;
      switch (b) {
        case 0: av = a11 + a22; bv = b11 + b22; break;
        case 1: av = a21 + a22; bv = b11; break;
        case 2: av = a11; bv = b12 - b22; break;
        case 3: av = a22; bv = b21 - b11; break;
        case 4: av = a11 + a12; bv = b22; break;
        case 5: av = a21 - a11; bv = b11 + b12; break;
        default: av = a12 - a22; bv = b21 + b22; break;
      }
      long idx = (long)b * me + (long)r + (long)c * m;
      double ga = __half2float(hAP[idx]), gb = __half2float(hBP[idx]);
      if (fabs(av - ga) > pa[b]) pa[b] = fabs(av - ga);
      if (fabs(bv - gb) > pb[b]) pb[b] = fabs(bv - gb);
    }
    printf("  packA errs:"); for (int b = 0; b < 7; b++) printf(" %.4f", pa[b]); printf("\n");
    printf("  packB errs:"); for (int b = 0; b < 7; b++) printf(" %.4f", pb[b]); printf("\n");
    free(hAP); free(hBP);
    // (b) verify the 7 products
    __half* hMP = (__half*)malloc((size_t)7 * me * 2);
    CKU(cudaMemcpy(hMP, MP, (size_t)7 * me * 2, cudaMemcpyDeviceToHost));
    double em[7] = {0,0,0,0,0,0,0};
    for (int b = 0; b < 7; b++) for (int c = 0; c < m; c++) for (int r = 0; r < m; r++) {
      double s = 0;
      for (int t = 0; t < m; t++) {
        long ai = (long)r + (long)t * n, bi = (long)t + (long)c * n;      // a11
        long aj = (long)r + (long)(t + m) * n, bj = (long)(t + m) + (long)c * n; // a12/b21
        long ak = (long)(r + m) + (long)t * n, bk = (long)t + (long)(c + m) * n; // a21/b12
        long al = (long)(r + m) + (long)(t + m) * n, bl = (long)(t + m) + (long)(c + m) * n; // a22
        double a11 = __half2float(hA[ai]), a12 = __half2float(hA[aj]);
        double a21 = __half2float(hA[ak]), a22 = __half2float(hA[al]);
        double b11 = __half2float(hB[bi]), b21 = __half2float(hB[bj]);
        double b12 = __half2float(hB[bk]), b22 = __half2float(hB[bl]);
        double av, bv;
        switch (b) {
          case 0: av = a11 + a22; bv = b11 + b22; break;
          case 1: av = a21 + a22; bv = b11; break;
          case 2: av = a11; bv = b12 - b22; break;
          case 3: av = a22; bv = b21 - b11; break;
          case 4: av = a11 + a12; bv = b22; break;
          case 5: av = a21 - a11; bv = b11 + b12; break;
          default: av = a12 - a22; bv = b21 + b22; break;
        }
        s += av * bv;
      }
      double got = __half2float(hMP[(long)b * me + (long)r + (long)c * m]);
      if (fabs(s - got) > em[b]) em[b] = fabs(s - got);
    }
    printf("  M errs:"); for (int b = 0; b < 7; b++) printf(" %.4f", em[b]); printf("\n");
    free(hMP);
  }

  cudaEvent_t s, e; CKU(cudaEventCreate(&s)); CKU(cudaEventCreate(&e));
  double tf[2], tmin[2];
  for (int which = 0; which < 2; which++) {
    for (int i = 0; i < 2; i++) { if (which) run_strassen(); else gemm(h, n, A, B, C1, 1); }
    CKU(cudaDeviceSynchronize());
    double sum = 0; tmin[which] = 1e9;
    for (int i = 0; i < reps; i++) {
      CKU(cudaEventRecord(s));
      if (which) run_strassen(); else gemm(h, n, A, B, C1, 1);
      CKU(cudaEventRecord(e)); CKU(cudaEventSynchronize(e));
      float ms; CKU(cudaEventElapsedTime(&ms, s, e));
      sum += ms;
      if (ms < tmin[which]) tmin[which] = ms;
    }
    tf[which] = sum / reps;
  }
  double fl = 2.0 * n * n * n;
  printf("n=%d  cuBLAS avg %.3f min %.3f ms (min %.1f TFLOPS) | fused-Strassen avg %.3f min %.3f ms (min %.1f TFLOPS)\n",
         n, tf[0], tmin[0], fl / (tmin[0] * 1e9), tf[1], tmin[1], fl / (tmin[1] * 1e9));
  printf("   ratio avg %.3f  |  ratio min %.3f\n", tf[0] / tf[1], tmin[0] / tmin[1]);
  return 0;
}
