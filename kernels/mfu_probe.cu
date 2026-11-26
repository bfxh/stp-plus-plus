// MFU probe: what can this card ACTUALLY do on the shapes a 1B model uses?
// Sweeps M (batch*seq) for the three real GEMM families of Llama-3.2-1B:
//   qkv/o : (M, 2048, 2048)
//   gate/up: (M, 8192, 2048)
//   down  : (M, 2048, 8192)
//   lm_head: (M, 32000, 2048)
// Reports best-of-N TFLOPS so a background training job cannot fake the number.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cublasStatus_t s_ = (cublasStatus_t)(x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  printf("CUBLAS ERR line %d code %d\n", __LINE__, (int)s_); exit(1);} } while (0)
#define CKU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA ERR line %d : %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static float bench(cublasHandle_t h, int M, int N, int K,
                   const __half* A, const __half* B, __half* C, int reps) {
  float al = 1.f, be = 0.f;
  cudaEvent_t s, e; CKU(cudaEventCreate(&s)); CKU(cudaEventCreate(&e));
  float best = 1e9f;
  for (int r = 0; r < reps; r++) {
    CKU(cudaEventRecord(s));
    CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al,
                    B, CUDA_R_16F, N, A, CUDA_R_16F, K, &be,
                    C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CKU(cudaEventRecord(e)); CKU(cudaEventSynchronize(e));
    float ms; CKU(cudaEventElapsedTime(&ms, s, e));
    if (ms < best) best = ms;
  }
  cudaEventDestroy(s); cudaEventDestroy(e);
  return best;   // ms
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 30;
  cublasHandle_t h; CK(cublasCreate(&h));

  // largest buffers: M=8192 x K=8192, N=32000 x K=2048 -> alloc generously
  const int MAXM = 8192, MAXN = 32000, MAXK = 8192;
  __half *A = nullptr, *B = nullptr, *C = nullptr;
  CKU(cudaMalloc(&A, (size_t)MAXM * MAXK * 2));
  CKU(cudaMalloc(&B, (size_t)MAXN * MAXK * 2));
  CKU(cudaMalloc(&C, (size_t)MAXM * MAXN * 2));
  // fill with something non-degenerate (avoids denormal slowdowns)
  CKU(cudaMemset(A, 0x3c, (size_t)MAXM * MAXK * 2));
  CKU(cudaMemset(B, 0x3c, (size_t)MAXN * MAXK * 2));

  printf("%-22s %8s %8s %10s %10s %8s\n", "shape (M,N,K)", "ms", "TFLOPS", "peak88", "MFU%", "sol%");
  struct Shape { int M, N, K; const char* tag; };
  std::vector<Shape> shapes;
  for (int M : {1024, 2048, 4096, 8192}) {
    shapes.push_back({M, 2048, 2048,  "qkv/o"});
    shapes.push_back({M, 8192, 2048,  "gate/up"});
    shapes.push_back({M, 2048, 8192,  "down"});
    shapes.push_back({M, 4096, 8192,  "mlp wide"});
  }
  shapes.push_back({1024, 32000, 2048, "lm_head"});

  double peak = 88.0;  // TFLOPS, FP16 tensor on RTX 4060 Ti (to be confirmed by best shape)
  double best_tf = 0;
  for (auto& sh : shapes) {
    float ms = bench(h, sh.M, sh.N, sh.K, A, B, C, reps);
    double tf = 2.0 * sh.M * sh.N * sh.K / (ms * 1e9);
    if (tf > best_tf) best_tf = tf;
    printf("%-6s (%5d,%5d,%5d) %8.3f %8.1f %9.1f%% %9.1f%%\n", sh.tag, sh.M, sh.N, sh.K,
           ms, tf, tf / peak * 100, tf / peak * 100);
  }
  printf("\nbest observed = %.1f TFLOPS  -> implied realistic peak >= %.1f\n", best_tf, best_tf);
  return 0;
}
