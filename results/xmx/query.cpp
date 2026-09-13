#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl::ext::oneapi::experimental::matrix;
int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "device: " << dev.get_info<sycl::info::device::name>() << "\n";
    auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    std::cout << "joint_matrix combinations: " << combos.size() << "\n";
    auto tn = [](matrix_type t){ switch(t){ case matrix_type::sint8: return "s8"; case matrix_type::uint8: return "u8"; case matrix_type::fp16: return "f16"; case matrix_type::bf16: return "bf16"; case matrix_type::fp32: return "f32"; case matrix_type::sint32: return "s32"; case matrix_type::tf32: return "tf32"; default: return "?"; } };
    for (auto & c : combos) {
        std::cout << "  A=" << tn(c.atype) << " B=" << tn(c.btype) << " C=" << tn(c.ctype) << " D=" << tn(c.dtype)
                  << "  M=" << c.msize << " N=" << c.nsize << " K=" << c.ksize
                  << "  maxM=" << c.max_msize << " maxN=" << c.max_nsize << " maxK=" << c.max_ksize << "\n";
    }
    std::cout << "sub-group sizes:";
    for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) std::cout << " " << s;
    std::cout << "\nlocal mem: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes\n";
    return 0;
}
