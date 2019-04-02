#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

at::Tensor d_sigmoid(at::Tensor z) {auto s = at::sigmoid(z);return (1 - s) * s;}

template <typename scalar_t>
__device__ __forceinline__ scalar_t hard_sigmoid(scalar_t z) {
  if(z>=1.0 || z<=0.0)
  {
      return (z>=1.0)?1.0:0.0;
  }
  else
  {
      return z;
  }
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_hard_sigmoid(scalar_t z) {
  return (z>=1 || z<=0)?0:1.0;
}
template <typename scalar_t> __global__ void hard_sigmoid_kernel(
    const scalar_t * __restrict__ x,
    scalar_t * __restrict__ h,
    size_t column_max) 
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * column_max + column;
    if (column < column_max) {
        h[index] = hard_sigmoid(x[index]);
    }
}
template < typename scalar_t > __global__ void d_hard_sigmoid_kernel(
    const scalar_t * __restrict__ x,
    scalar_t * __restrict__ d_h,
    size_t column_max) 
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * column_max + column;
    if (column < column_max) {
        d_h[index] = d_hard_sigmoid(x[index]);
    }
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t sgn(scalar_t z) {
  return (z>=1.0)?1.0:-1.0;
}
template <typename scalar_t> __global__ void sgn_kernel(
    const scalar_t * __restrict__ x,
    scalar_t * __restrict__ result,
    size_t column_max) 
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * column_max + column;
    if (column < column_max) {
        result[index] = sgn(x[index]);
    }
}
template <typename scalar_t> __device__ __forceinline__ scalar_t noisy_hard_sigmoid(scalar_t x,scalar_t alpha,
                                                                                    scalar_t c,scalar_t p,
                                                                                    scalar_t alpha_complement,
                                                                                    scalar_t sgn,scalar_t random)
{
    if(x<=1 && x>=0)
    {
        return x;
    }
    else
    {
        h =  hard_sigmoid(x);
        scalar_t slope = x>=1?:1:-1;
        scalar_t d = slope*sgn;
        scalar_t diff = x - h;
        scalar_t sigma = at::pow((at::sigmoid(p*diff)*c-0.5),2);
        return alpha*h+(1-alpha)*x+d*sigma*random;
    }
}
template <typename scalar_t> __device__ __forceinline__ scalar_t d_noisy_hard_sigmoid(scalar_t d_x,
                                                                                      scalar_t x,
                                                                                      scalar_t alpha,
                                                                                      scalar_t c,scalar_t p,
                                                                                      scalar_t alpha_complement,
                                                                                      scalar_t sgn,scalar_t random)
{
    if(x<=1 && x>=0)
    {
        return 1;
    }
    else
    {
        scalar_t diff = x - h;
        scalar_t p_diff = p*diff;
        scalar_t d_diff = p*d_p_diff*c*2*(at::sigmoid(p_diff)-0.5)*d*z*random;
        scalar_t d_h = z*alpha+d_diff;
        return at::sigmoid(x)*x*d_h+(1-alpha);
    }
}

std::vector<at::Tensor> noisy_sigmoid_cuda_forward(
    at::Tensor x,
    at::Tensor alpha,
    at::Tensor c,
    at::Tensor p,
    at::Tensor alpha_complement,
    at::Tensor alpha_complement_sgn,
    at::Tensor random)
{
    const auto batch_size = x.size(0);
    const auto state_size = x.size(1);
    auto h = at::zeros_like(x);
    auto value_sgn = at::zeros_like(x);
    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);
    AT_DISPATCH_FLOATING_TYPES(pre_gate_i.type(), "hard_sigmoid_kernel", ([&] {
        hard_sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            x.data<scalar_t>(),
            h.data<scalar_t>(),
            state_size);
        }
    ));
    AT_DISPATCH_FLOATING_TYPES(pre_gate_i.type(), "sgn_kernel", ([&] {
        sgn_kernel<scalar_t><<<blocks, threads>>>(
            x.data<scalar_t>(),
            value_sgn.data<scalar_t>(),
            state_size);
        }
    ));
    __syncthreads();
    auto diff = h-x;
    auto d = alpha_complement_sgn*value_sgn;
    auto p_diff = p*diff;
    auto diff_coef = at::sigmoid(p_diff)*c-0.5;
    auto sigma = at::pow(diff_coef,2);
    auto result = a;pha*h+alpha_complement*x+sigma*random;
    return {result,diff_coef,p_diff};
}

std::vector<at::Tensor> noisy_sigmoid_cuda_backward(
    at::Tensor d_y,
    at::Tensor diff_coef,
    at::Tensor p_diff,
    at::Tensor d,
    at::Tensor x,
    at::Tensor alpha,
    at::Tensor c,
    at::Tensor p,
    at::Tensor alpha_complement,
    at::Tensor alpha_complement_sgn,
    at::Tensor random
)
{
    auto d_sigma = random*d_y*d;
    auto d_diff_coef = d_sigma*2*diff_coef;
    auto d_p_diff = d_sigmoid(p_diff);
    auto d_diff = c*p*d_p_diff*d_diff_coef;
    const auto batch_size = x.size(0);
    const auto state_size = x.size(1);
    auto d_x_h_part_temp = at::zeros_like(x);
    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);
    auto temp = alpha*d_y + d_diff;
    AT_DISPATCH_FLOATING_TYPES(pre_gate_i.type(), "d_hard_sigmoid_kernel", ([&] {
        d_hard_sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            temp.data<scalar_t>(),
            d_x_h_part_temp.data<scalar_t>(),
            state_size);
        }
    ));
    __syncthreads();
    auto d_x = -d_diff + d_x_h_part_temp + d_y*alpha_complement;
    return {d_x};
}