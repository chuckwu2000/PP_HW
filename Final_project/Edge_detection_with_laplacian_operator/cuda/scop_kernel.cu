#include "scop_kernel.hu"
__global__ void kernel0(double *buffer, int *in)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 5996)
      for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 5996); c3 += 16)
        buffer[192000 * b0 + 32 * b1 + 6000 * t0 + c3 + 6001] = abs((8 * in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3 + 1)]) - (((((((in[(32 * b0 + t0) * 6000 + (32 * b1 + c3)] + in[(32 * b0 + t0) * 6000 + (32 * b1 + c3 + 1)]) + in[(32 * b0 + t0) * 6000 + (32 * b1 + c3 + 2)]) + in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3)]) + in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3 + 2)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3 + 1)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3 + 2)]));
}
__global__ void kernel1(double *buffer, double max, double min, int *out)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    double private_val;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    if (32 * b0 + t0 <= 5999)
      for (int c1 = 32 * ((72 * b0 + b1 + 256) % 256) + 96000 * b0; c1 <= ppcg_min(18002999, 96000 * b0 + 98999); c1 += 8192)
        for (int c3 = ppcg_max(t1, 96000 * b0 + 2992 * t0 + t1 + 16 * ppcg_fdiv_q(8 * t0 - t1 - c1 - 1, 16) + 16); c3 <= ppcg_min(31, 96000 * b0 + 3000 * t0 - c1 + 5999); c3 += 16) {
          private_val = ((255 * (buffer[96000 * b0 + 3000 * t0 + c1 + c3] - min)) / (max - min));
          if (private_val > 15) {
            private_val = 255;
          }
          out[(32 * b0 + t0) * 6000 + (-96000 * b0 - 3000 * t0 + c1 + c3)] = private_val;
        }
}
