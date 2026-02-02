#include <torch/extension.h>

extern "C" void solve(
  const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

torch::Tensor attention_cuda_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
  // Ensure inputs are contiguous
  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  // Get dimensions
  int M = Q.size(0);
  int d = Q.size(1);
  int N = K.size(0);

  // Basic checks
  TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

  TORCH_CHECK(K.size(1) == d, "K feature dim must match Q");
  TORCH_CHECK(V.size(0) == N, "V sequence length must match K");
  TORCH_CHECK(V.size(1) == d, "V feature dim must match d");

  auto output = torch::empty({M, d}, torch::dtype(torch::kFloat32).device(Q.device()));

  solve(Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M,
        N,
        d);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &attention_cuda_forward, "Hand-written SoftmaxAttention forward (CUDA)");
}
