#include "gpu_scop_kernel.hu"
__global__ void kernel0(double *buffer, int *in)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 5996)
      for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 5996); c3 += 16)
        buffer[192000 * b0 + 32 * b1 + 6000 * t0 + c3 + 6001] = abs((8 * in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3 + 1)]) - (((((((in[(32 * b0 + t0) * 6000 + (32 * b1 + c3)] + in[(32 * b0 + t0) * 6000 + (32 * b1 + c3 + 1)]) + in[(32 * b0 + t0) * 6000 + (32 * b1 + c3 + 2)]) + in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3)]) + in[(32 * b0 + t0 + 1) * 6000 + (32 * b1 + c3 + 2)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3 + 1)]) + in[(32 * b0 + t0 + 2) * 6000 + (32 * b1 + c3 + 2)]));
}
