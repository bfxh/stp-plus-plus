// Clean Strassen (one level) vs cuBLAS, pure C++/CUDA. Blocks are packed into
// contiguous m*m buffers (honest: the packing copies show up in the timing).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %d @%d\n",(int)s,__LINE__);exit(1);} } while(0)

__global__ void addsub_k(__half* dst, const __half* a, const __half* b, int numel, float sign) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) dst[i] = __float2half(__half2float(a[i]) + sign * __half2float(b[i]));
}
static void addsub(__half* dst, const __half* a, const __half* b, int numel, float sign) {
  int t = 256, g = (numel + t - 1) / t;
  addsub_k<<<g, t>>>(dst, a, b, numel, sign); CK(cudaGetLastError());
}
static void gemm(cublasHandle_t h, int n, const __half* A, const __half* B, __half* C) {
  float al = 1.f, be = 0.f;
  CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, A, CUDA_R_16F, n,
                  B, CUDA_R_16F, n, &be, C, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
static void pack(__half* dst, const __half* src, int m, int ld, int bi, int bj) {
  CK(cudaMemcpy2DAsync(dst, m * 2, src + (size_t)bi * m + (size_t)bj * m * ld, ld * 2,
                  m * 2, m, cudaMemcpyDeviceToDevice, 0));
}
static void unpack(__half* dst, const __half* src, int m, int ld, int bi, int bj) {
  CK(cudaMemcpy2DAsync(dst + (size_t)bi * m + (size_t)bj * m * ld, ld * 2, src, m * 2,
                       m * 2, m, cudaMemcpyDeviceToDevice, 0));
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2048;
  int iters = argc > 2 ? atoi(argv[2]) : 10;
  int m = n / 2;
  size_t ne = (size_t)n * n, me = (size_t)m * m;
  cublasHandle_t h; CB(cublasCreate(&h));

  std::vector<__half> hA(ne), hB(ne);
  unsigned s = 7;
  for (size_t i = 0; i < ne; ++i) { s = s*1664525u+1013904223u; hA[i] = __float2half((float)(s>>8)/16777216.f - .5f); }
  for (size_t i = 0; i < ne; ++i) { s = s*1664525u+1013904223u; hB[i] = __float2half((float)(s>>8)/16777216.f - .5f); }

  __half *A, *B, *C1, *C2, *WS;
  CK(cudaMalloc(&A, ne*2)); CK(cudaMalloc(&B, ne*2)); CK(cudaMalloc(&C1, ne*2));
  CK(cudaMalloc(&C2, ne*2)); CK(cudaMalloc(&WS, 27 * me * 2));
  CK(cudaMemcpy(A, hA.data(), ne*2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(B, hB.data(), ne*2, cudaMemcpyHostToDevice));

  __half* A11=WS+0*me;  __half* A12=WS+1*me;  __half* A21=WS+2*me;  __half* A22=WS+3*me;
  __half* B11=WS+4*me;  __half* B12=WS+5*me;  __half* B21=WS+6*me;  __half* B22=WS+7*me;
  __half* T1=WS+8*me;   __half* T2=WS+9*me;   __half* T3=WS+10*me;  __half* T4=WS+11*me;
  __half* T5=WS+12*me;  __half* T6=WS+13*me;  __half* T7=WS+14*me;  __half* T8=WS+15*me;
  __half* T9=WS+16*me;  __half* T10=WS+17*me;
  __half* M1=WS+18*me;  __half* M2=WS+19*me;  __half* M3=WS+20*me;  __half* M4=WS+21*me;
  __half* M5=WS+22*me;  __half* M6=WS+23*me;  __half* M7=WS+24*me;
  __half* O11=WS+25*me; __half* O12=WS+25*me; __half* O21=WS+26*me; __half* O22=WS+26*me;

  auto run = [&]() {
    pack(A11,A,m,n,0,0); pack(A12,A,m,n,0,1); pack(A21,A,m,n,1,0); pack(A22,A,m,n,1,1);
    pack(B11,B,m,n,0,0); pack(B12,B,m,n,0,1); pack(B21,B,m,n,1,0); pack(B22,B,m,n,1,1);
    addsub(T1,A11,A22,m*m,+1); addsub(T2,B11,B22,m*m,+1); gemm(h,m,T1,T2,M1);
    addsub(T3,A21,A22,m*m,+1); gemm(h,m,T3,B11,M2);
    addsub(T4,B12,B22,m*m,-1); gemm(h,m,A11,T4,M3);
    addsub(T5,B21,B11,m*m,-1); gemm(h,m,A22,T5,M4);
    addsub(T6,A11,A12,m*m,+1); gemm(h,m,T6,B22,M5);
    addsub(T7,A21,A11,m*m,-1); addsub(T8,B11,B12,m*m,+1); gemm(h,m,T7,T8,M6);
    addsub(T9,A12,A22,m*m,-1); addsub(T10,B21,B22,m*m,+1); gemm(h,m,T9,T10,M7);
    addsub(T1,M1,M4,m*m,+1); addsub(T1,T1,M5,m*m,-1); addsub(T1,T1,M7,m*m,+1);
    addsub(T2,M3,M5,m*m,+1);
    addsub(T3,M2,M4,m*m,+1);
    addsub(T4,M1,M3,m*m,+1); addsub(T4,T4,M2,m*m,-1); addsub(T4,T4,M6,m*m,+1);
    unpack(C2,T1,m,n,0,0); unpack(C2,T2,m,n,0,1); unpack(C2,T3,m,n,1,0); unpack(C2,T4,m,n,1,1);
  };

  run(); CK(cudaDeviceSynchronize());
  auto chk = [&](const __half* dev, int bi, int bj, const std::vector<__half>& src) {
    std::vector<__half> tmp(me);
    CK(cudaMemcpy(tmp.data(), dev, me * 2, cudaMemcpyDeviceToHost));
    double e = 0;
    for (int j = 0; j < m; ++j) for (int i = 0; i < m; ++i) {
      double want = __half2float(src[(size_t)(bi*m+i) + (size_t)(bj*m+j)*n]);
      e = fmax(e, fabs(__half2float(tmp[i + j*m]) - want));
    }
    return e;
  };
  printf("packA errs %.3f %.3f %.3f %.3f | packB %.3f %.3f %.3f %.3f\n",
         chk(A11,0,0,hA), chk(A12,0,1,hA), chk(A21,1,0,hA), chk(A22,1,1,hA),
         chk(B11,0,0,hB), chk(B12,0,1,hB), chk(B21,1,0,hB), chk(B22,1,1,hB));
  gemm(h,n,A,B,C1); CK(cudaDeviceSynchronize());

  // per-M check against a CPU product of the *packed* blocks
  auto dump = [&](const __half* d) { std::vector<__half> t(me); CK(cudaMemcpy(t.data(), d, me*2, cudaMemcpyDeviceToHost)); return t; };
  auto cp = [&](std::vector<float>& out, const std::vector<__half>& X, const std::vector<__half>& Y) {
    out.assign(me, 0.f);
    for (int j = 0; j < m; ++j) for (int k = 0; k < m; ++k) {
      float b = __half2float(Y[k + j*m]);
      for (int i = 0; i < m; ++i) out[i + j*m] += __half2float(X[i + k*m]) * b;
    }
  };
  auto sub = [&](const std::vector<__half>& X, const std::vector<__half>& Y, float s) {
    std::vector<float> o(me);
    for (size_t i = 0; i < me; ++i) o[i] = __half2float(X[i]) + s*__half2float(Y[i]);
    return o;
  };
  auto errM = [&](const __half* g, const std::vector<float>& r) {
    std::vector<__half> t = dump(g); double e = 0;
    for (size_t i = 0; i < me; ++i) e = fmax(e, fabs(__half2float(t[i]) - r[i]));
    return e;
  };
  std::vector<__half> a11 = dump(A11), a12 = dump(A12), a21 = dump(A21), a22 = dump(A22);
  std::vector<__half> b11 = dump(B11), b12 = dump(B12), b21 = dump(B21), b22 = dump(B22);
  auto toH = [&](const std::vector<float>& f) { std::vector<__half> o(me); for (size_t i=0;i<me;++i) o[i]=__float2half(f[i]); return o; };
  std::vector<float> pM1, pM2, pM3, pM4, pM5, pM6, pM7;
  cp(pM1, toH(sub(a11,a22,+1)), toH(sub(b11,b22,+1)));
  cp(pM2, toH(sub(a21,a22,+1)), b11);
  cp(pM3, a11, toH(sub(b12,b22,-1)));
  cp(pM4, a22, toH(sub(b21,b11,-1)));
  cp(pM5, toH(sub(a11,a12,+1)), b22);
  cp(pM6, toH(sub(a21,a11,-1)), toH(sub(b11,b12,+1)));
  cp(pM7, toH(sub(a12,a22,-1)), toH(sub(b21,b22,+1)));
  // expected C11 = M1+M4-M5+M7, compare with T1 (holds C11 after run())
  {
    std::vector<float> e(me);
    for (size_t i=0;i<me;++i) e[i] = pM1[i]+pM4[i]-pM5[i]+pM7[i];
    printf("err C11(vs T1)=%.4f  ", errM(T1,e));
    std::vector<float> e2(me);
    for (size_t i=0;i<me;++i) e2[i] = pM1[i]+pM3[i]-pM2[i]+pM6[i];
    printf("err C22(vs T4)=%.4f\n", errM(T4,e2));
  }

  std::vector<__half> x1(ne), x2(ne);
  CK(cudaMemcpy(x1.data(),C1,ne*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(x2.data(),C2,ne*2,cudaMemcpyDeviceToHost));
  double e=0,mx=0; size_t arg=0;
  for (size_t i=0;i<ne;++i){ double a=__half2float(x1[i]),b=__half2float(x2[i]);
    if (fabs(a-b)>e){e=fabs(a-b);arg=i;}
    mx=fmax(mx,fabs(a)); }
  printf("n=%d  |C|max=%.3f  max|strassen-cublas|=%.4f at (r=%zu,c=%zu) cublas=%.3f str=%.3f\n",
         n,mx,e,arg%n,arg/n,__half2float(x1[arg]),__half2float(x2[arg]));
  // does the unpack land where we think?
  for (int q = 0; q < 4; ++q) {
    const __half* Tq = (q == 0) ? T1 : (q == 1) ? T2 : (q == 2) ? T3 : T4;
    int bi = q / 2, bj = q % 2;
    std::vector<__half> tq = dump(Tq);
    double e = 0;
    for (int j = 0; j < m; ++j) for (int i = 0; i < m; ++i)
      e = fmax(e, fabs(__half2float(tq[i + j*m]) -
                      __half2float(x2[(size_t)(bi*m+i) + (size_t)(bj*m+j)*n])));
    printf("quad%d(bi=%d,bj=%d) unpack err=%.4f\n", q, bi, bj, e);
  }

  auto tm = [&](bool str) {
    for(int i=0;i<3;++i) str?run():gemm(h,n,A,B,C1);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<iters;++i) str?run():gemm(h,n,A,B,C1);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/iters;
  };
  double d=tm(false), st=tm(true), fl=2.0*n*n*n;
  printf("n=%d  cuBLAS %.3f ms (%.1f TFLOPS) | Strassen %.3f ms (%.1f TFLOPS) | ratio %.3f\n",
         n,d,fl/d/1e9,st,fl/st/1e9,d/st);
  return 0;
}
