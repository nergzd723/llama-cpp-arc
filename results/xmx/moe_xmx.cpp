// Standalone prototype: MoE GEMM on Intel XMX (joint_matrix fp16 -> fp32) with the quantized
// expert weights (IQ2_S or IQ4_NL) decoded in-kernel into a packed local-memory tile that is
// shared by 32 activation rows. Checks a few outputs against a CPU reference and times the
// model's real shapes. Build: icpx -fsycl -O3 moe_xmx.cpp -o moe_xmx
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"   // block_iq2_s, block_iq4_nl, iq2s_grid, kvalues_iq4nl, kmask_iq2xs

using namespace sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;

static constexpr int TM = 32;   // activation rows per work-group (4 sub-groups x 8)
static constexpr int TN = 64;   // weight rows per work-group (4 B tiles x 16)
static constexpr int TK = 32;   // K elements decoded per step (one IQ4_NL block, one IQ2_S sub-block)
static constexpr int SG = 16;   // sub-group size required by XMX
static constexpr int NSG = 4;   // sub-groups per work-group

struct tile_desc { int32_t expert; int32_t row0; }; // padded activation row range [row0, row0+TM) of one expert

// decode 32 consecutive elements [k0, k0+32) of weight row w of expert block base
template <typename block_t> struct decoder;
template <> struct decoder<block_iq4_nl> {
    static constexpr int K_BLOCK = 32;
    static inline void decode32(const block_iq4_nl * row_blocks, int k0, half * out, const int8_t * kvals) {
        const block_iq4_nl & b = row_blocks[k0 / 32];
        const float d = (float) b.d;
        for (int j = 0; j < 16; j++) {
            out[j]      = (half) (d * kvals[b.qs[j] & 0xf]);
            out[j + 16] = (half) (d * kvals[b.qs[j] >> 4]);
        }
    }
};
template <> struct decoder<block_iq2_s> {
    static constexpr int K_BLOCK = 256;
    static inline void decode32(const block_iq2_s * row_blocks, int k0, half * out, const uint64_t * grid) {
        const block_iq2_s & b = row_blocks[k0 / 256];
        const int ib = (k0 % 256) / 32;             // 32-sub-block index 0..7
        const float d = (float) b.d;
        const uint8_t sc = b.scales[ib];
        for (int il = 0; il < 4; il++) {           // 4 groups of 8
            const uint8_t * g = (const uint8_t *) (grid + (b.qs[4*ib + il] | ((b.qh[ib] << (8 - 2*il)) & 0x300)));
            const float dl = d * (0.5f + ((sc >> (4 * (il / 2))) & 0xf)) * 0.25f;
            const uint8_t signs = b.qs[32 + 4*ib + il];
            for (int j = 0; j < 8; j++) {
                out[8*il + j] = (half) (dl * g[j] * ((signs & (1u << j)) ? -1.f : 1.f));
            }
        }
    }
};

template <typename block_t>
static void run_kernel(sycl::queue & q, const block_t * W, const half * A, float * C, const tile_desc * tiles, int n_tiles,
                       int M, int K, size_t expert_stride_bytes, const void * table) {
    const int n_tiles_n = M / TN;
    const int blocks_per_row = K / decoder<block_t>::K_BLOCK;
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<half, 1> Bs(TN * TK, h); // packed: (TK/2) rows of TN*2
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tiles, n_tiles_n * NSG * SG), sycl::range<2>(1, NSG * SG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int t = it.get_group(0);
                const int tn = it.get_group(1);
                const tile_desc td = tiles[t];
                sycl::sub_group sg = it.get_sub_group();
                const int sgid = it.get_local_id(1) / SG;
                const int lid  = it.get_local_id(1);

                const block_t * Wexp = (const block_t *) ((const char *) W + (size_t) td.expert * expert_stride_bytes);
                const half * Arow = A + (size_t) (td.row0 + sgid * 8) * K;

                joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16> acc[4];
                for (int j = 0; j < 4; j++) joint_matrix_fill(sg, acc[j], 0.0f);

                half * bs = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
                for (int k0 = 0; k0 < K; k0 += TK) {
                    // cooperative decode: thread lid decodes weight row tn*TN + lid, elements [k0, k0+32)
                    {
                        half vals[TK];
                        const block_t * row_blocks = Wexp + (size_t) (tn * TN + lid) * blocks_per_row;
                        if constexpr (std::is_same_v<block_t, block_iq2_s>) {
                            decoder<block_t>::decode32(row_blocks, k0, vals, (const uint64_t *) table);
                        } else {
                            decoder<block_t>::decode32(row_blocks, k0, vals, (const int8_t *) table);
                        }
                        // packed layout: element (k, n) at [(k/2) * (TN*2) + n*2 + (k%2)]
                        for (int k = 0; k < TK; k++) {
                            bs[(k / 2) * (TN * 2) + lid * 2 + (k % 2)] = vals[k];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int s = 0; s < TK / 16; s++) {
                        joint_matrix<sycl::sub_group, half, use::a, 8, 16, layout::row_major> a;
                        joint_matrix_load(sg, a, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Arow + k0 + 16 * s), K);
                        for (int j = 0; j < 4; j++) {
                            joint_matrix<sycl::sub_group, half, use::b, 16, 16, layout::ext_intel_packed> b;
                            joint_matrix_load(sg, b, Bs.get_multi_ptr<sycl::access::decorated::no>() + (8 * s) * (TN * 2) + (16 * j) * 2, TN * 2);
                            joint_matrix_mad(sg, acc[j], a, b, acc[j]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                float * Crow = C + (size_t) (td.row0 + sgid * 8) * M + tn * TN;
                for (int j = 0; j < 4; j++) {
                    joint_matrix_store(sg, acc[j], sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Crow + 16 * j), M, layout::row_major);
                }
            });
    });
}

template <typename block_t>
static void bench(sycl::queue & q, const char * name, int M, int K, int n_experts, int rows_per_expert, int reps) {
    const int blocks_per_row = K / decoder<block_t>::K_BLOCK;
    const size_t expert_bytes = (size_t) M * blocks_per_row * sizeof(block_t);
    std::mt19937 rng(42);
    // random but valid blocks
    std::vector<uint8_t> Wh(expert_bytes * n_experts);
    for (auto & x : Wh) x = (uint8_t) rng();
    {   // sane scales: small d
        for (size_t e = 0; e < (size_t) n_experts; e++) for (size_t b = 0; b < (size_t) M * blocks_per_row; b++) {
            block_t * blk = (block_t *) (Wh.data() + e * expert_bytes) + b;
            { const sycl::half hd = (sycl::half) (0.01f + 0.02f * (rng() % 100) / 100.f); memcpy(&blk->d, &hd, sizeof(hd)); }
        }
    }
    // padded rows: each expert gets rows_per_expert rows, padded to TM
    const int rows_pad = ((rows_per_expert + TM - 1) / TM) * TM;
    const size_t n_rows_total = (size_t) rows_pad * n_experts;
    std::vector<half> Ah(n_rows_total * K);
    for (auto & x : Ah) x = (half) (((int) (rng() % 200) - 100) / 100.f);
    std::vector<tile_desc> tiles;
    for (int e = 0; e < n_experts; e++) for (int r = 0; r < rows_pad; r += TM) tiles.push_back({e, (int32_t) (e * rows_pad + r)});

    block_t * W = (block_t *) sycl::malloc_device(Wh.size(), q);
    half * A = sycl::malloc_device<half>(Ah.size(), q);
    float * C = sycl::malloc_device<float>(n_rows_total * M, q);
    tile_desc * T = sycl::malloc_device<tile_desc>(tiles.size(), q);
    q.memcpy(W, Wh.data(), Wh.size()).wait();
    q.memcpy(A, Ah.data(), Ah.size() * sizeof(half)).wait();
    q.memcpy(T, tiles.data(), tiles.size() * sizeof(tile_desc)).wait();
    void * table;
    if constexpr (std::is_same_v<block_t, block_iq2_s>) { table = sycl::malloc_device(sizeof(iq2s_grid), q); q.memcpy(table, iq2s_grid, sizeof(iq2s_grid)).wait(); }
    else { table = sycl::malloc_device(sizeof(kvalues_iq4nl), q); q.memcpy(table, kvalues_iq4nl, sizeof(kvalues_iq4nl)).wait(); }

    run_kernel<block_t>(q, W, A, C, T, (int) tiles.size(), M, K, expert_bytes, table); q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) run_kernel<block_t>(q, W, A, C, T, (int) tiles.size(), M, K, expert_bytes, table);
    q.wait();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;

    // reference check on a few (row, col) samples
    std::vector<float> Ch(n_rows_total * M);
    q.memcpy(Ch.data(), C, Ch.size() * sizeof(float)).wait();
    double max_rel = 0; int checked = 0;
    for (int s = 0; s < 64; s++) {
        const int e = rng() % n_experts, r = rng() % rows_per_expert, m = rng() % M;
        const size_t row = (size_t) e * rows_pad + r;
        const block_t * rb = (const block_t *) (Wh.data() + (size_t) e * expert_bytes) + (size_t) m * blocks_per_row;
        double ref = 0, norm = 0; half tmp[32];
        for (int k0 = 0; k0 < K; k0 += 32) {
            if constexpr (std::is_same_v<block_t, block_iq2_s>) decoder<block_t>::decode32(rb, k0, tmp, iq2s_grid);
            else decoder<block_t>::decode32(rb, k0, tmp, kvalues_iq4nl);
            for (int k = 0; k < 32; k++) { const double w = (float) tmp[k], a = (float) Ah[row * K + k0 + k]; ref += w * a; norm += std::fabs(w * a); }
        }
        const double got = Ch[row * M + m];
        const double rel = std::fabs(got - ref) / (norm + 1e-6);
        if (s < 2) printf("   sample: got %.6f ref %.6f norm %.4f\n", got, ref, norm);
        max_rel = std::max(max_rel, rel); checked++;
    }
    const double flops = 2.0 * (double) rows_per_expert * n_experts * M * K;
    printf("%-8s M=%5d K=%5d experts=%d rows/expert=%3d (padded %3d): %8.2f ms/op  %7.2f TFLOPS(useful)  max rel err %.2e over %d samples\n",
           name, M, K, n_experts, rows_per_expert, rows_pad, ms, flops / ms / 1e9, max_rel, checked);
    sycl::free(W, q); sycl::free(A, q); sycl::free(C, q); sycl::free(T, q); sycl::free(table, q);
}

int main(int argc, char ** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    // the model's shapes: gate/up IQ2_S [2560 -> 640], down IQ4_NL [640 -> 2560]; ~464 used experts
    for (int rows : {10, 40, 96}) {
        bench<block_iq2_s>(q, "iq2_s", 640, 2560, 464, rows, reps);
        bench<block_iq4_nl>(q, "iq4_nl", 2560, 640, 464, rows, reps);
    }
    return 0;
}
