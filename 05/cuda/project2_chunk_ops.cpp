#include <torch/extension.h>

#include <cstdint>

torch::Tensor trim_chunks_base65536_cuda(torch::Tensor chunks);
int64_t compare_abs_base65536_cuda(torch::Tensor left, torch::Tensor right);
torch::Tensor add_abs_base65536_cuda(torch::Tensor left, torch::Tensor right);
torch::Tensor sub_abs_base65536_cuda(torch::Tensor left, torch::Tensor right);
torch::Tensor mul_small_base65536_cuda(torch::Tensor chunks, int64_t multiplier);
torch::Tensor div2_base65536_cuda(torch::Tensor chunks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trim_chunks_base65536", &trim_chunks_base65536_cuda, "Trim a little-endian base-2^16 chunk tensor on CUDA");
    m.def("compare_abs_base65536", &compare_abs_base65536_cuda, "Compare two little-endian base-2^16 chunk tensors on CUDA");
    m.def("add_abs_base65536", &add_abs_base65536_cuda, "Add two little-endian base-2^16 chunk tensors on CUDA");
    m.def("sub_abs_base65536", &sub_abs_base65536_cuda, "Subtract two little-endian base-2^16 chunk tensors on CUDA");
    m.def("mul_small_base65536", &mul_small_base65536_cuda, "Multiply a base-2^16 chunk tensor by a small scalar on CUDA");
    m.def("div2_base65536", &div2_base65536_cuda, "Divide a base-2^16 chunk tensor by 2 on CUDA");
}
