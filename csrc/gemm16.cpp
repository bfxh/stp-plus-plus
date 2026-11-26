// Zero-copy fp16-accumulate GEMM (CUBLAS_COMPUTE_16F) for Ada.
// Build with g++ only (no nvcc, no device code) against the CUDA 12.8 headers/libs
// that torch already ships.  All three entry points write straight into a
// row-major buffer, so the autograd path never materialises a transpose.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

static cublasHandle_t handle() {
  static cublasHandle_t h = nullptr;
  if (!h) {
    TORCH_CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
  }
  cublasSetStream(h, at::cuda::getCurrentCUDAStream());
  return h;
}

static void chk2(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuda tensors required");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "fp16 required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous required");
}

static void run(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                const void* A, int lda, const void* B, int ldb, void* C, int ldc) {
  __half al = __float2half(1.f), be = __float2half(0.f);
  TORCH_CHECK(cublasGemmEx(handle(), ta, tb, m, n, k, &al, A, CUDA_R_16F, lda,
                           B, CUDA_R_16F, ldb, &be, C, CUDA_R_16F, ldc,
                           CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
              == CUBLAS_STATUS_SUCCESS, "cublasGemmEx(16F) failed");
}

// forward: y = x @ w^T ,  x (M,K), w (N,K)  ->  y (M,N)
torch::Tensor nt(torch::Tensor x, torch::Tensor w) {
  chk2(x, w);
  int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  TORCH_CHECK((int)w.size(1) == K, "nt: K mismatch");
  auto y = torch::empty({M, N}, x.options());
  run(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
      w.data_ptr(), K, x.data_ptr(), K, y.data_ptr(), N);
  return y;
}

// dx = g @ w ,  g (M,N), w (N,K)  ->  dx (M,K)
torch::Tensor dx(torch::Tensor g, torch::Tensor w) {
  chk2(g, w);
  int M = (int)g.size(0), N = (int)g.size(1), K = (int)w.size(1);
  TORCH_CHECK((int)w.size(0) == N, "dx: N mismatch");
  auto d = torch::empty({M, K}, g.options());
  run(CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
      w.data_ptr(), K, g.data_ptr(), N, d.data_ptr(), K);
  return d;
}

// dw = g^T @ x ,  g (M,N), x (M,K)  ->  dw (N,K)   (no transpose materialised)
torch::Tensor dw(torch::Tensor g, torch::Tensor x) {
  chk2(g, x);
  int M = (int)g.size(0), N = (int)g.size(1), K = (int)x.size(1);
  TORCH_CHECK((int)x.size(0) == M, "dw: M mismatch");
  auto d = torch::empty({N, K}, g.options());
  run(CUBLAS_OP_N, CUBLAS_OP_T, K, N, M,
      x.data_ptr(), K, g.data_ptr(), N, d.data_ptr(), K);
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nt", &nt, "y = x @ w^T (fp16 accumulate)");
  m.def("dx", &dx, "dx = g @ w (fp16 accumulate)");
  m.def("dw", &dw, "dw = g^T @ x (fp16 accumulate)");
}
