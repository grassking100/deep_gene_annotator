#include <torch/extension.h>
#include <vector>
// CUDA forward declarations
std::vector<at::Tensor> noisy_sigmoid_cuda_forward(
    at::Tensor x,
    at::Tensor alpha,
    at::Tensor c,
    at::Tensor p,
    at::Tensor alpha_complement,
    at::Tensor alpha_complement_sgn,
    at::Tensor random
);

std::vector<at::Tensor> noisy_sigmoid_cuda_backward(
    at::Tensor grad_x,
    at::Tensor diff_coef,
    at::Tensor x,
    at::Tensor alpha,
    at::Tensor c,
    at::Tensor p,
    at::Tensor alpha_complement,
    at::Tensor alpha_complement_sgn,
    at::Tensor random
);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> noisy_sigmoid_forward(
    at::Tensor x,
    at::Tensor alpha,
    at::Tensor c,
    at::Tensor p,
    at::Tensor alpha_complement,
    at::Tensor alpha_complement_sgn,
    at::Tensor random
){
  CHECK_INPUT(x);
  CHECK_INPUT(alpha);
  CHECK_INPUT(c);
  CHECK_INPUT(p);
  CHECK_INPUT(weights_i);
  CHECK_INPUT(alpha_complement);
  CHECK_INPUT(alpha_complement_sgn);
  CHECK_INPUT(random);

  return noisy_sigmoid_cuda_forward(x,alpha,c,p,alpha_complement,alpha_complement_sgn,random);
}

std::vector<at::Tensor> noisy_sigmoid_backward(
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
) {
  CHECK_INPUT(d_y);
  CHECK_INPUT(diff_coef);
  CHECK_INPUT(p_diff);
  CHECK_INPUT(d);
  CHECK_INPUT(x);
  CHECK_INPUT(alpha);
  CHECK_INPUT(c);
  CHECK_INPUT(p);
  CHECK_INPUT(weights_i);
  CHECK_INPUT(alpha_complement);
  CHECK_INPUT(alpha_complement_sgn);
  CHECK_INPUT(random);
  return noisy_sigmoid_cuda_backward(d_y,diff_coef,p_diff,d,x,alpha,c,p,alpha_complement,alpha_complement_sgn,random);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &noisy_sigmoid_forward, "Noisy Hard Activation function forward (CUDA)");
  m.def("backward", &noisy_sigmoid_backward, "Noisy Hard Activation function backward (CUDA)");
}
