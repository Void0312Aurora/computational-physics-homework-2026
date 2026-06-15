#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace {

constexpr int64_t kBase = 1ll << 16;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 1024;

struct CarryState {
    int64_t g;
    int64_t p;
};

struct CarryCompose {
    __host__ __device__ CarryState operator()(const CarryState& prefix, const CarryState& next) const {
        return CarryState{
            next.g | (next.p & prefix.g),
            next.p & prefix.p,
        };
    }
};

void check_cuda_int64_1d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt64, name, " must have dtype int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void effective_length_kernel(const int64_t* values, int64_t length, int64_t* out_length) {
    __shared__ int64_t shared_max[kThreadsPerBlock];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    int64_t local_max = 0;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        if (values[index] != 0) {
            local_max = index + 1;
        }
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_max[threadIdx.x] =
                shared_max[threadIdx.x] > shared_max[threadIdx.x + offset]
                    ? shared_max[threadIdx.x]
                    : shared_max[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(out_length), static_cast<unsigned long long>(shared_max[0]));
    }
}

__global__ void compare_abs_kernel(
    const int64_t* left,
    int64_t left_length,
    const int64_t* right,
    int64_t* out_packed
) {
    __shared__ int64_t shared_index[kThreadsPerBlock];
    __shared__ int64_t shared_sign[kThreadsPerBlock];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    int64_t local_index = -1;
    int64_t local_sign = 0;

    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < left_length; index += stride) {
        const int64_t lhs = left[index];
        const int64_t rhs = right[index];
        if (lhs != rhs && index > local_index) {
            local_index = index;
            local_sign = lhs > rhs ? 1 : -1;
        }
    }
    shared_index[threadIdx.x] = local_index;
    shared_sign[threadIdx.x] = local_sign;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset && shared_index[threadIdx.x + offset] > shared_index[threadIdx.x]) {
            shared_index[threadIdx.x] = shared_index[threadIdx.x + offset];
            shared_sign[threadIdx.x] = shared_sign[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && shared_index[0] >= 0) {
        const unsigned long long packed =
            (static_cast<unsigned long long>(shared_index[0] + 1) << 1) |
            static_cast<unsigned long long>(shared_sign[0] > 0 ? 1 : 0);
        atomicMax(reinterpret_cast<unsigned long long*>(out_packed), packed);
    }
}

__global__ void add_abs_kernel(
    const int64_t* left,
    int64_t left_length,
    const int64_t* right,
    int64_t right_length,
    int64_t* raw_sum,
    CarryState* states
) {
    const int64_t max_length = left_length > right_length ? left_length : right_length;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < max_length; index += stride) {
        int64_t sum = 0;
        if (index < left_length) {
            sum += left[index];
        }
        if (index < right_length) {
            sum += right[index];
        }
        raw_sum[index] = sum;
        states[index] = CarryState{
            sum >= kBase ? 1 : 0,
            sum == (kBase - 1) ? 1 : 0,
        };
    }
}

__global__ void add_finalize_kernel(
    const int64_t* raw_sum,
    const CarryState* prefix_states,
    int64_t length,
    int64_t* out,
    int64_t* out_carry
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const int64_t carry_in = prefix_states[index].g;
        const int64_t total = raw_sum[index] + carry_in;
        out[index] = total & (kBase - 1);
        if (index + 1 == length) {
            const int64_t carry = total >> 16;
            *out_carry = carry;
            if (carry != 0) {
                out[length] = carry;
            }
        }
    }
}

__global__ void sub_abs_kernel(
    const int64_t* left,
    int64_t left_length,
    const int64_t* right,
    int64_t right_length,
    int64_t* raw_diff,
    CarryState* states
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < left_length; index += stride) {
        int64_t diff = left[index];
        if (index < right_length) {
            diff -= right[index];
        }
        raw_diff[index] = diff;
        states[index] = CarryState{
            diff < 0 ? 1 : 0,
            diff == 0 ? 1 : 0,
        };
    }
}

__global__ void sub_finalize_kernel(
    const int64_t* raw_diff,
    const CarryState* prefix_states,
    int64_t length,
    int64_t* out,
    int64_t* out_status
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const int64_t borrow_in = prefix_states[index].g;
        const int64_t total = raw_diff[index] - borrow_in;
        out[index] = total < 0 ? total + kBase : total;
        if (index + 1 == length) {
            *out_status = total < 0 ? -1 : 0;
        }
    }
}

__global__ void mul_small_kernel(
    const int64_t* chunks,
    int64_t length,
    int64_t multiplier,
    int64_t* out
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const unsigned long long product =
            static_cast<unsigned long long>(chunks[index]) * static_cast<unsigned long long>(multiplier);
        out[index] = static_cast<int64_t>(product);
    }
}

__global__ void carry_extract_kernel(
    int64_t* values,
    int64_t length,
    int64_t* carry,
    int64_t* nonzero_flag
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const int64_t value = values[index];
        const int64_t next_carry = value >> 16;
        values[index] = value & (kBase - 1);
        carry[index] = next_carry;
        if (next_carry != 0) {
            atomicMax(reinterpret_cast<unsigned long long*>(nonzero_flag), static_cast<unsigned long long>(1));
        }
    }
}

__global__ void carry_shift_add_kernel(
    int64_t* values,
    const int64_t* carry,
    int64_t length
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const int64_t carry_in = index == 0 ? 0 : carry[index - 1];
        values[index] += carry_in;
    }
}

__global__ void div2_kernel(const int64_t* chunks, int64_t length, int64_t* out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < length; index += stride) {
        const int64_t next = index + 1 < length ? chunks[index + 1] : 0;
        out[index] = (chunks[index] >> 1) + ((next & 1) << 15);
    }
}

int64_t fetch_scalar_int64(const torch::Tensor& device_scalar) {
    auto cpu_value = device_scalar.to(torch::kCPU);
    return cpu_value.item<int64_t>();
}

int compute_block_count(int64_t length) {
    int64_t blocks = (length + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::max<int64_t>(1, std::min<int64_t>(blocks, kMaxBlocks));
    return static_cast<int>(blocks);
}

int64_t effective_length_cuda(const torch::Tensor& chunks) {
    auto out_length = torch::zeros({1}, chunks.options());
    effective_length_kernel<<<compute_block_count(chunks.size(0)), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
        chunks.data_ptr<int64_t>(),
        chunks.size(0),
        out_length.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::max<int64_t>(1, fetch_scalar_int64(out_length));
}

void exclusive_scan_states(torch::Tensor states, torch::Tensor prefix_states, cudaStream_t stream) {
    const auto* input_ptr = reinterpret_cast<const CarryState*>(states.data_ptr<int64_t>());
    auto* output_ptr = reinterpret_cast<CarryState*>(prefix_states.data_ptr<int64_t>());
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        input_ptr,
        input_ptr + states.size(0),
        output_ptr,
        CarryState{0, 1},
        CarryCompose{}
    );
}

torch::Tensor narrow_trimmed(const torch::Tensor& values, int64_t length) {
    if (length <= 0) {
        return torch::zeros({1}, values.options());
    }
    if (length >= values.size(0)) {
        return values;
    }
    return values.narrow(0, 0, length);
}

}  // namespace

torch::Tensor trim_chunks_base65536_cuda(torch::Tensor chunks) {
    check_cuda_int64_1d(chunks, "chunks");
    int64_t length = effective_length_cuda(chunks);
    return narrow_trimmed(chunks, length);
}

int64_t compare_abs_base65536_cuda(torch::Tensor left, torch::Tensor right) {
    check_cuda_int64_1d(left, "left");
    check_cuda_int64_1d(right, "right");
    int64_t left_length = effective_length_cuda(left);
    int64_t right_length = effective_length_cuda(right);
    if (left_length > right_length) {
        return 1;
    }
    if (left_length < right_length) {
        return -1;
    }
    auto out_packed = torch::zeros({1}, left.options());
    compare_abs_kernel<<<compute_block_count(left_length), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
        left.data_ptr<int64_t>(),
        left_length,
        right.data_ptr<int64_t>(),
        out_packed.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t packed = fetch_scalar_int64(out_packed);
    if (packed == 0) {
        return 0;
    }
    return (packed & 1) != 0 ? 1 : -1;
}

torch::Tensor add_abs_base65536_cuda(torch::Tensor left, torch::Tensor right) {
    check_cuda_int64_1d(left, "left");
    check_cuda_int64_1d(right, "right");
    int64_t left_length = effective_length_cuda(left);
    int64_t right_length = effective_length_cuda(right);
    const int64_t max_length = std::max(left_length, right_length);
    auto raw_sum = torch::zeros({max_length}, left.options());
    auto states = torch::empty({max_length, 2}, left.options());
    auto prefix_states = torch::empty({max_length, 2}, left.options());
    auto out = torch::zeros({max_length + 1}, left.options());
    auto out_carry = torch::zeros({1}, left.options());
    auto stream = at::cuda::getDefaultCUDAStream();
    add_abs_kernel<<<compute_block_count(max_length), kThreadsPerBlock, 0, stream>>>(
        left.data_ptr<int64_t>(),
        left_length,
        right.data_ptr<int64_t>(),
        right_length,
        raw_sum.data_ptr<int64_t>(),
        reinterpret_cast<CarryState*>(states.data_ptr<int64_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    exclusive_scan_states(states, prefix_states, stream);
    add_finalize_kernel<<<compute_block_count(max_length), kThreadsPerBlock, 0, stream>>>(
        raw_sum.data_ptr<int64_t>(),
        reinterpret_cast<CarryState*>(prefix_states.data_ptr<int64_t>()),
        max_length,
        out.data_ptr<int64_t>(),
        out_carry.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t carry = fetch_scalar_int64(out_carry);
    return narrow_trimmed(out, max_length + (carry != 0 ? 1 : 0));
}

torch::Tensor sub_abs_base65536_cuda(torch::Tensor left, torch::Tensor right) {
    check_cuda_int64_1d(left, "left");
    check_cuda_int64_1d(right, "right");
    int64_t left_length = effective_length_cuda(left);
    int64_t right_length = effective_length_cuda(right);
    auto out = torch::zeros({left_length}, left.options());
    auto raw_diff = torch::zeros({left_length}, left.options());
    auto states = torch::empty({left_length, 2}, left.options());
    auto prefix_states = torch::empty({left_length, 2}, left.options());
    auto out_status = torch::zeros({1}, left.options());
    auto stream = at::cuda::getDefaultCUDAStream();
    sub_abs_kernel<<<compute_block_count(left_length), kThreadsPerBlock, 0, stream>>>(
        left.data_ptr<int64_t>(),
        left_length,
        right.data_ptr<int64_t>(),
        right_length,
        raw_diff.data_ptr<int64_t>(),
        reinterpret_cast<CarryState*>(states.data_ptr<int64_t>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    exclusive_scan_states(states, prefix_states, stream);
    sub_finalize_kernel<<<compute_block_count(left_length), kThreadsPerBlock, 0, stream>>>(
        raw_diff.data_ptr<int64_t>(),
        reinterpret_cast<CarryState*>(prefix_states.data_ptr<int64_t>()),
        left_length,
        out.data_ptr<int64_t>(),
        out_status.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    TORCH_CHECK(fetch_scalar_int64(out_status) == 0, "sub_abs_base65536_cuda requires left >= right");
    return narrow_trimmed(out, effective_length_cuda(out));
}

torch::Tensor mul_small_base65536_cuda(torch::Tensor chunks, int64_t multiplier) {
    check_cuda_int64_1d(chunks, "chunks");
    TORCH_CHECK(multiplier >= 0, "multiplier must be non-negative");
    TORCH_CHECK(
        multiplier <= std::numeric_limits<int64_t>::max() / (kBase - 1),
        "multiplier is too large for the current int64 mul_small kernel"
    );
    int64_t length = effective_length_cuda(chunks);
    const int64_t total_length = length + 8;
    auto out = torch::zeros({total_length}, chunks.options());
    auto carry = torch::zeros({total_length}, chunks.options());
    auto has_carry = torch::zeros({1}, chunks.options());
    auto stream = at::cuda::getDefaultCUDAStream();
    mul_small_kernel<<<compute_block_count(length), kThreadsPerBlock, 0, stream>>>(
        chunks.data_ptr<int64_t>(),
        length,
        multiplier,
        out.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int kCarryPassesPerSync = 2;
    bool needs_another_pass = false;
    do {
        for (int pass = 0; pass < kCarryPassesPerSync; ++pass) {
            has_carry.zero_();
            carry_extract_kernel<<<compute_block_count(total_length), kThreadsPerBlock, 0, stream>>>(
                out.data_ptr<int64_t>(),
                total_length,
                carry.data_ptr<int64_t>(),
                has_carry.data_ptr<int64_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            carry_shift_add_kernel<<<compute_block_count(total_length), kThreadsPerBlock, 0, stream>>>(
                out.data_ptr<int64_t>(),
                carry.data_ptr<int64_t>(),
                total_length
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        needs_another_pass = fetch_scalar_int64(has_carry) != 0;
    } while (needs_another_pass);

    return narrow_trimmed(out, effective_length_cuda(out));
}

torch::Tensor div2_base65536_cuda(torch::Tensor chunks) {
    check_cuda_int64_1d(chunks, "chunks");
    int64_t length = effective_length_cuda(chunks);
    auto out = torch::zeros({length}, chunks.options());
    div2_kernel<<<compute_block_count(length), kThreadsPerBlock, 0, at::cuda::getDefaultCUDAStream()>>>(
        chunks.data_ptr<int64_t>(),
        length,
        out.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return narrow_trimmed(out, effective_length_cuda(out));
}
