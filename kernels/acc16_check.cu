// Is CUBLAS_COMPUTE_16F numerically safe at K=2048 for this model's real value ranges?
// All matrices column-major (cublas native): C = A(MxK) * B(KxN).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CKU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA ERR %s\n", cudaGetErrorString(e_)); exit(1);} } while (0)
#define CK(x) do { cublasStatus_t s_ = (cublasStatus_t)(x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS ERR %d\n", (int)s_); exit(1);} } while (0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int N = argc > 2 ? atoi(argv[2]) : 8192;
  int K = argc > 3 ? atoi(argv[3]) : 2048;
  double sd_a = argc > 4 ? atof(argv[4]) : 1.0;
  double sd_b = argc > 5 ? atof(argv[5]) : 0.02;

  __half *A, *B, *C16, *C32; __half *hA, *hB, *h16, *h32;
  CKU(cudaMalloc(&A, (size_t)M * K * 2)); CKU(cudaMalloc(&B, (size_t)K * N * 2));
  CKU(cudaMalloc(&C16, (size_t)M * N * 2)); CKU(cudaMalloc(&C32, (size_t)M * N * 2));
  hA = (__half*)malloc((size_t)M * K * 2); hB = (__half*)malloc((size_t)K * N * 2);
  h16 = (__half*)malloc((size_t)M * N * 2); h32 = (__half*)malloc((size_t)M * N * 2);

  std::mt19937 g(7); std::normal_distribution<double> da(0.0, sd_a), db(0.0, sd_b);
  for (long i = 0; i < (long)M * K; i++) hA[i] = __float2half((float)da(g));   // A[r + t*M]
  for (long i = 0; i < (long)K * N; i++) hB[i] = __float2half((float)db(g));   // B[t + c*K]
  CKU(cudaMemcpy(A, hA, (size_t)M * K * 2, cudaMemcpyHostToDevice));
  CKU(cudaMemcpy(B, hB, (size_t)K * N * 2, cudaMemcpyHostToDevice));

  cublasHandle_t h; CK(cublasCreate(&h));
  float al = 1.f, be = 0.f; __half a16 = __float2half(1.f), b16 = __float2half(0.f);
  CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &al, A, CUDA_R_16F, M,
                  B, CUDA_R_16F, K, &be, C32, CUDA_R_16F, M, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &a16, A, CUDA_R_16F, M,
                  B, CUDA_R_16F, K, &b16, C16, CUDA_R_16F, M, CUBLAS_COMPUTE_16F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CKU(cudaDeviceSynchronize());
  CKU(cudaMemcpy(h16, C16, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
  CKU(cudaMemcpy(h32, C32, (size_t)M * N * 2, cudaMemcpyDeviceToHost));

  int ns = 64; double rms = 0, d32 = 0, d16 = 0, worst16 = 0;
  for (int s = 0; s < ns; s++) {
    int r = (int)((double)rand() / RAND_MAX * M), c = (int)((double)rand() / RAND_MAX * N);
    double ref = 0;
    for (int t = 0; t < K; t++)
      ref += (double)__half2float(hA[(long)r + (long)t * M]) * (double)__half2float(hB[(long)t + (long)c * K]);
    double v32 = __half2float(h32[(long)r + (long)c * M]);
    double v16 = __half2float(h16[(long)r + (long)c * M]);
    rms += ref * ref;
    double e32 = fabs(v32 - ref), e16 = fabs(v16 - ref);
    if (e32 > d32) d32 = e32;
    if (e16 > d16) d16 = e16;
    if (e16 / (fabs(ref) + 1e-3) > worst16) worst16 = e16 / (fabs(ref) + 1e-3);
  }
  rms = sqrt(rms / ns);
  long nbad = 0;
  for (long i = 0; i < (long)M * N; i++) if (!std::isfinite(__half2float(h16[i]))) nbad++;
  printf("M=%d N=%d K=%d  act_sd=%.2f w_sd=%.3f  |C|rms=%.3f\n", M, N, K, sd_a, sd_b, rms);
  printf("  fp32-acc : max abs err %.3e  (= %.2f%% of rms)\n", d32, d32 / rms * 100);
  printf("  fp16-acc : max abs err %.3e  (= %.2f%% of rms)   worst rel %.2f%%   non-finite %ld\n",
         d16, d16 / rms * 100, worst16 * 100, nbad);
  printf("  verdict  : %s\n", (d16 / rms < 0.02 && nbad == 0) ? "USABLE (fp16-acc error < 2% of rms)" : "RISKY");
  return 0;
}
