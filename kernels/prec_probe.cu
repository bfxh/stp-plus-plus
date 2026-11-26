// Which precision/accumulate mode is actually fastest on this card, at our real shapes?
// Tests: FP16 in + FP32 acc (baseline), FP16 in + FP16 acc, FP8(E4M3) in + FP32 acc, BF16 in + FP32 acc.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define TRY(x) do { cublasStatus_t s_ = (cublasStatus_t)(x); if (s_ != CUBLAS_STATUS_SUCCESS) { return -1; } } while (0)
#define CKU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA ERR %s\n", cudaGetErrorString(e_)); exit(1);} } while (0)

static float t(cublasHandle_t h, int mode, int M, int N, int K, void* A, void* B, void* C, int reps) {
  float al = 1.f, be = 0.f; __half a16 = __float2half(1.f), b16 = __float2half(0.f);
  cudaEvent_t s, e; CKU(cudaEventCreate(&s)); CKU(cudaEventCreate(&e));
  float best = 1e9f;
  for (int r = 0; r < reps; r++) {
    CKU(cudaEventRecord(s));
    int st;
    if (mode == 0)      st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &be, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    else if (mode == 1) st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a16, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &b16, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    else if (mode == 2) st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_8F_E4M3, N, A, CUDA_R_8F_E4M3, K, &be, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    else                st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return -1.f;
    CKU(cudaEventRecord(e)); CKU(cudaEventSynchronize(e));
    float ms; CKU(cudaEventElapsedTime(&ms, s, e));
    if (ms < best) best = ms;
  }
  cudaEventDestroy(s); cudaEventDestroy(e);
  return best;
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 20;
  cublasHandle_t h; if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
  const int MAXM = 8192, MAXN = 8192, MAXK = 8192;
  void *A, *B, *C, *A8, *B8, *Abf, *Bbf, *Cbf;
  CKU(cudaMalloc(&A, (size_t)MAXM * MAXK * 2)); CKU(cudaMalloc(&B, (size_t)MAXN * MAXK * 2)); CKU(cudaMalloc(&C, (size_t)MAXM * MAXN * 2));
  CKU(cudaMalloc(&A8, (size_t)MAXM * MAXK));    CKU(cudaMalloc(&B8, (size_t)MAXN * MAXK));
  CKU(cudaMalloc(&Abf, (size_t)MAXM * MAXK * 2)); CKU(cudaMalloc(&Bbf, (size_t)MAXN * MAXK * 2)); CKU(cudaMalloc(&Cbf, (size_t)MAXM * MAXN * 2));
  CKU(cudaMemset(A, 0x3c, (size_t)MAXM * MAXK * 2)); CKU(cudaMemset(B, 0x3c, (size_t)MAXN * MAXK * 2));
  CKU(cudaMemset(A8, 0x38, (size_t)MAXM * MAXK));    CKU(cudaMemset(B8, 0x38, (size_t)MAXN * MAXK));
  CKU(cudaMemset(Abf, 0x3f, (size_t)MAXM * MAXK * 2)); CKU(cudaMemset(Bbf, 0x3f, (size_t)MAXN * MAXK * 2));

  const char* names[4] = {"fp16/acc32", "fp16/acc16", "fp8e4m3", "bf16/acc32"};
  int Ms[4] = {1024, 4096, 8192, 8192};
  int Ns[4] = {8192, 8192, 8192, 8192};
  int Ks[4] = {2048, 2048, 2048, 2048};

  printf("%-28s %8s %9s %9s\n", "shape / mode", "ms", "TFLOPS", "vs fp16acc32");
  for (int i = 0; i < 4; i++) {
    float base = t(h, 0, Ms[i], Ns[i], Ks[i], A, B, C, reps);
    double fl = 2.0 * Ms[i] * Ns[i] * Ks[i];
    printf("(%5d,%5d,%5d)\n", Ms[i], Ns[i], Ks[i]);
    for (int m = 0; m < 4; m++) {
      void *pa = A, *pb = B, *pc = C;
      if (m == 2) { pa = A8; pb = B8; }
      if (m == 3) { pa = Abf; pb = Bbf; pc = Cbf; }
      float ms = t(h, m, Ms[i], Ns[i], Ks[i], pa, pb, pc, reps);
      if (ms < 0) { printf("  %-26s  UNSUPPORTED\n", names[m]); continue; }
      double tf = fl / (ms * 1e9);
      printf("  %-26s %8.3f %9.1f %8.3fx\n", names[m], ms, tf, base / ms);
    }
  }
  return 0;
}
