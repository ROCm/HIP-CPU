/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

constexpr auto niter{10000000};
const Approx PI{std::atan(1) * 4};

int do_on_cpu()
{
    using namespace std;

    minstd_rand gen{random_device{}()};
    uniform_real_distribution<float> dist;

    double x,y;
    int count = 0;
    double z;

    for (auto i = 0; i != niter; ++i) {
        x = dist(gen);
        y = dist(gen);
        z = (x * x) + (y * y);
        if (z <= 1) ++count;
    }

    return count;
}

__global__
void pi_kernel(int* count_d)
{
    using namespace std;

    const auto elts_per_lane =
        (niter + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    const auto tid = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

    minstd_rand gen{random_device{}()};
    uniform_real_distribution<float> dist;

    for (auto i = 0u; i != elts_per_lane; ++i) {
        double x, y, z;

        x = dist(gen);
        y = dist(gen);
        z = (x * x) + (y * y);

        if (tid + i >= niter) return;

        count_d[tid + i] = z <= 1 ? 1 : 0;
    }
}

TEST_CASE("Monte-Carlo PI")
{
    using namespace std;

    int n{INT_MIN};
    BENCHMARK("CPU") { n = do_on_cpu(); };

    CHECK(PI == (static_cast<double>(n) / niter) * 4.0);

    unsigned int threads = 16;
    unsigned int blocks = thread::hardware_concurrency();

    int* count_d;
    int* count = (int*)malloc(niter * sizeof(int));
    hipMalloc((void**)&count_d, niter *  sizeof(int));

    BENCHMARK("HIP-CPU") {
        hipLaunchKernelGGL(
            pi_kernel, dim3{blocks}, dim3{threads}, 0, nullptr, count_d);
        hipDeviceSynchronize();
    };

    hipMemcpy(count, count_d, niter * sizeof(int), hipMemcpyDeviceToHost);

    n = reduce(execution::par_unseq, count, count + niter, 0);

    CHECK(PI == (static_cast<double>(n) / niter) * 4.0);

    hipFree(count_d);
    free(count);
}

void add(int n, float* x, float* y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

__global__
void kernel_add(int n, float* x, float* y)
{
    const auto elts_per_lane{
        (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x)};
    const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * elts_per_lane;
    const auto cnt{std::min(elts_per_lane, n - index)};

    for (auto i = 0u; i != cnt; ++i) {
        y[index + i] = x[index + i] + y[index + i];
    }
}

TEST_CASE("VADD")
{
    using namespace std;

    constexpr auto N{1 << 29};

    auto x = new float[N];
    auto y = new float[N];

    fill_n(x, N, 1.0f);
    fill_n(y, N, 2.0f);

    BENCHMARK("CPU") { return add(N, x, y); };

    auto blockSize{1024};
    auto numBlocks{thread::hardware_concurrency()};

    BENCHMARK("HIP-CPU") {
        hipLaunchKernelGGL(
            kernel_add, dim3(numBlocks), dim3(blockSize), 0, nullptr, N, x, y);
        hipDeviceSynchronize();
    };

    delete [] x;
    delete [] y;
}

void matrixMultiplication_cpu(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int N)
{
    for (auto k = 0; k != N; ++k) {
        for (auto i = 0; i != N; ++i) {
            const auto r = A[i * N + k];

            for (auto j = 0; j != N; ++j) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

template<typename T>
__global__
void matrixMultiplicationKernel(
    const T* __restrict A,
    const T* __restrict B,
    T* __restrict C,
    unsigned int N)
{
    const auto row_stride{gridDim.y * blockDim.y};
    const auto col_stride{gridDim.x * blockDim.x};

    for (auto ROW = blockIdx.y * blockDim.y + threadIdx.y; ROW < N; ROW += row_stride) {
        for (auto COL = blockIdx.x * blockDim.x + threadIdx.x; COL < N; COL += col_stride) {
            T tmpSum{0};
            for (auto i = 0u; i < N; ++i) {
                tmpSum += A[ROW * N + i] * B[i * N + COL];
            }
            C[ROW * N + COL] = tmpSum;
        }
    }
}


template<unsigned int BLOCK_SIZE, typename T>
__global__
void matrixMultiplication_hip_cpu(
    const T* __restrict A,
    const T* __restrict B,
    T* C,
    int n)
{
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    T tmp = 0;
    int idx;

    for (auto sub = 0u; sub != gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n * n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = A[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = B[idx];
        }
        __syncthreads();

        for (auto k = 0u; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        C[row * n + col] = tmp;
    }
}

template<unsigned int TS, typename T>
__global__
void MatrixMultiplyTiled(
    const T* __restrict A,
    const T* __restrict B,
    T* __restrict C,
    unsigned int N)
{
    const auto row{threadIdx.x};
    const auto col{threadIdx.y};
    const auto global_x{blockIdx.x * blockDim.x + row};
    const auto global_y{blockIdx.y * blockDim.y + col};

   for (auto gidx = global_x; gidx < N; gidx += gridDim.x * blockDim.x) {
       for (auto gidy = global_y; gidy < N; gidy += gridDim.y * blockDim.y) {
            T sum{0};
            for (auto i = 0u; i < N; i += TS) {
                __shared__ T locA[TS][TS];
                __shared__ T locB[TS][TS];

                locA[row][col] = A[gidx * N + col + i];
                locB[row][col] = B[(row + i) * N + gidy];

                __syncthreads();

                for (auto k = 0u; k != TS; ++k)
                    sum += locA[row][k] * locB[k][col];

                __syncthreads();
            }

            C[gidx * N + gidy] = sum;
       }
   }
}

constexpr unsigned int block_size{128u};

TEST_CASE("SGEMM")
{
    using namespace std;

    int N = 1024;
    int SIZE = N * N;

    // Allocate memory on the host
    vector<float> A(SIZE);
    vector<float> B(SIZE);
    vector<float> C(SIZE);
    vector<float> D(SIZE);

    // Initialize matrices on the host
    for (auto i = 0; i != N; ++i) {
        for (auto j = 0; j != N; ++j) {
            A[i * N + j] = static_cast<float>(sin(i));
            B[i * N + j] = static_cast<float>(cos(j));
        }
    }

    BENCHMARK("CPU") {
        return matrixMultiplication_cpu(data(A), data(B), data(C), N);
    };

    unsigned int grid_rows = (N + block_size - 1) / block_size;
    unsigned int grid_cols = (N + block_size - 1) / block_size;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(block_size, block_size);

    const auto pA{data(A)};
    const auto pB{data(B)};
    const auto pD{data(D)};
    BENCHMARK("HIP-CPU") {
        hipLaunchKernelGGL(
            //matrixMultiplication_hip_cpu<block_size>,
            //MatrixMultiplyTiled<block_size>,
            matrixMultiplicationKernel,
            dimGrid,
            dimBlock,
            0,
            nullptr,
            pA,
            pB,
            pD,
            N);
        return hipDeviceSynchronize();
    };
}

constexpr auto SOFTENING{1e-9f};

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float* data, int n)
{
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (std::rand() / static_cast<float>(RAND_MAX)) - 1.0f;
    }
}

void bodyForce(Body* p, float dt, int n)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx; p[i].vy += dt * Fy; p[i].vz += dt * Fz;
    }
}

__global__
void bodyForce_0(Body* p, float dt, unsigned int n)
{
    const auto elts_per_lane{
        (n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x)};
    const auto i{(blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane};

    if (i >= n) return;

    for (auto e = 0u; e != std::min(elts_per_lane, n - i); ++e) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (auto j = 0u; j < n; j++) {
            float dx = p[j].x - p[i + e].x;
            float dy = p[j].y - p[i + e].y;
            float dz = p[j].z - p[i + e].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i + e].vx += dt * Fx; p[i + e].vy += dt * Fy; p[i + e].vz += dt * Fz;
    }
}

__global__
void bodyForce_1(const float4* p, float4* v, float dt, unsigned int n)
{
    const auto elts_per_lane{
        (n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x)};
    const auto i{(blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane};

    if (i >= n) return;

    for (auto e = 0u; e != std::min(elts_per_lane, n - i); ++e) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (auto j = 0u; j < n; j++) {
            float dx = p[j].x - p[i + e].x;
            float dy = p[j].y - p[i + e].y;
            float dz = p[j].z - p[i + e].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        v[i + e].x += dt * Fx; v[i + e].y += dt * Fy; v[i + e].z += dt * Fz;
    }
}

constexpr auto BLOCK_SIZE{16};

template<unsigned int i, unsigned int n>
struct Unroller {
    template<typename F>
    __forceinline__
    static
    void body(F fn) noexcept
    {
        fn(i);
        Unroller<i + 1, n>::body(std::move(fn));
    }
};

template<unsigned int n>
struct Unroller<n, n> {
    template<typename F>
    __forceinline__
    static
    void body(F) noexcept {}
};

__global__
void bodyForce_2(const float4* p, float4* v, float dt, unsigned int n)
{
    const auto elts_per_lane{
        (n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x)};
    const auto i{(blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane};

    if (i >= n) return;

    for (auto e = 0u; e != std::min(elts_per_lane, n - i); ++e) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (auto tile = 0u; tile < gridDim.x; tile++) {
            __shared__ float3 spos[BLOCK_SIZE];
            float4 tpos = p[tile * blockDim.x + threadIdx.x];
            spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
            // Unroller<0, BLOCK_SIZE>::body([&](unsigned int j) noexcept __forceinline__ {
                float dx = spos[j].x - p[i + e].x;
                float dy = spos[j].y - p[i + e].y;
                float dz = spos[j].z - p[i + e].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = 1.0f / sqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
            }//);

            __syncthreads();
        }

        v[i + e].x += dt * Fx; v[i + e].y += dt * Fy; v[i + e].z += dt * Fz;
    }
}

TEST_CASE("N-Body")
{
    using namespace std;

    constexpr auto nBodies{30000};

    constexpr auto dt{0.01f};

    vector<Body> p(nBodies);
    auto buf{reinterpret_cast<float*>(data(p))};

    randomizeBodies(buf, 6 * nBodies);

    BENCHMARK("CPU") {
        bodyForce(data(p), dt, nBodies);

        for (auto i = 0; i != nBodies; ++i) {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    };

    const auto nBlocks{(nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE};

    randomizeBodies(buf, 6 * nBodies);

    BENCHMARK("HIP-CPU 0") {
        hipLaunchKernelGGL(
            bodyForce_0,
            dim3(nBlocks),
            dim3(BLOCK_SIZE),
            0,
            nullptr,
            reinterpret_cast<Body*>(buf),
            dt,
            nBodies);
        hipDeviceSynchronize();

        for (auto i = 0; i != nBodies; ++i) {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    };

    struct BodySystem { float4* pos, *vel; };
    vector<float4> p1(2 * nBodies);
    auto buf1{reinterpret_cast<float*>(data(p1))};

    randomizeBodies(buf1, 8 * nBodies);

    BodySystem bs{data(p1), data(p1) + nBodies};

    BENCHMARK("HIP-CPU 1") {
        hipLaunchKernelGGL(
            bodyForce_1,
            dim3(nBlocks),
            dim3(BLOCK_SIZE),
            0,
            nullptr,
            bs.pos,
            bs.vel,
            dt,
            nBodies);
        hipDeviceSynchronize();

        for (int i = 0 ; i < nBodies; i++) {
            bs.pos[i].x += bs.vel[i].x * dt;
            bs.pos[i].y += bs.vel[i].y * dt;
            bs.pos[i].z += bs.vel[i].z * dt;
        }
    };

    randomizeBodies(buf1, 8 * nBodies);

    BENCHMARK("HIP-CPU 2") {
        hipLaunchKernelGGL(
            bodyForce_2,
            dim3(nBlocks),
            dim3(BLOCK_SIZE),
            0,
            nullptr,
            bs.pos,
            bs.vel,
            dt,
            nBodies);
        hipDeviceSynchronize();

        for (int i = 0; i < nBodies; i++) {
            bs.pos[i].x += bs.vel[i].x * dt;
            bs.pos[i].y += bs.vel[i].y * dt;
            bs.pos[i].z += bs.vel[i].z * dt;
        }
    };
}

__host__ __device__
std::uint64_t solve(
    int N,
    int depth = 0,
    std::uint32_t left = 0,
    std::uint32_t mid = 0,
    std::uint32_t right = 0)
{
    if (depth == N) return 1;
    uint64_t sum = 0;
    for (auto pos = ((1u << N) - 1u) & ~(left | mid | right); pos; pos &= pos - 1) {
        std::uint32_t bit = pos & -pos;
        sum += solve(N, depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
    }

    return sum;
}

__global__
void kernel(
    int N,
    int depth,
    const uint32_t* const left_ary,
    const uint32_t* const mid_ary,
    const uint32_t* const right_ary,
    std::uint64_t* const result_ary,
    const std::size_t size)
{
    const auto elts_per_fiber{
        (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x)};
    const auto index = (threadIdx.x + blockIdx.x * blockDim.x) * elts_per_fiber;

    if (index >= size) return;

    for (auto i = 0u; i != std::min(elts_per_fiber, size - index); ++i) {
        result_ary[index + i] = solve(
            N,
            depth,
            left_ary[index + i],
            mid_ary[index + i],
            right_ary[index + i]);
    }
}

struct Node {
  int depth;
  std::uint32_t left, mid, right;
  std::uint32_t pos;
};

__device__
std::uint64_t solve_nonrec(
    int N,int depth, std::uint32_t left, std::uint32_t mid, std::uint32_t right)
{
    int stack_index = 0;
    std::uint64_t count = 0;
    __shared__ Node stack[64][32];
    stack[threadIdx.x][0] =
        {depth, left, mid, right, ((1u << N) - 1u) & ~(left | mid | right)};

    while (true) {
        if (stack[threadIdx.x][stack_index].depth == N) ++count;

        std::uint32_t pos = stack[threadIdx.x][stack_index].pos;
        if (pos == 0) {
            if (stack_index == 0) return count; // end solve
            --stack_index;
        }
        else {
            auto bit = pos & -pos;
            stack[threadIdx.x][stack_index].pos ^= bit;
            int new_depth = stack[threadIdx.x][stack_index].depth + 1;
            auto new_left = (stack[threadIdx.x][stack_index].left | bit) << 1;
            auto new_mid = stack[threadIdx.x][stack_index].mid | bit;
            auto new_right = (stack[threadIdx.x][stack_index].right | bit) >> 1;
            ++stack_index;
            stack[threadIdx.x][stack_index] = {
                new_depth,
                new_left,
                new_mid,
                new_right,
                ((1u << N) - 1u) & ~(new_left | new_mid | new_right)
            };
        }
    }
}

__global__
void kernel_ver2(
    int N,
    int depth,
    const std::uint32_t* const left_ary,
    const std::uint32_t* const mid_ary,
    const std::uint32_t* const right_ary,
    std::uint64_t* const result_ary,
    std::size_t size)
{
    const auto elts_per_fiber{
        (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x)};
    const auto index = (threadIdx.x + blockIdx.x * blockDim.x) * elts_per_fiber;

    if (index >= size) return;

    for (auto i = 0u; i != std::min(elts_per_fiber, size - index); ++i) {
        result_ary[index + i] = solve_nonrec(
            N,
            depth,
            left_ary[index + i],
            mid_ary[index + i],
            right_ary[index + i]);
    }
}

class NQueenExpand {
public:
    std::vector<std::uint32_t> left_ary, mid_ary, right_ary;
    void expand(
        int N,
        int M,
        int depth = 0,
        std::uint32_t left = 0,
        std::uint32_t mid = 0,
        std::uint32_t right = 0)
    {
        if (depth == M) {
            left_ary.push_back(left);
            mid_ary.push_back(mid);
            right_ary.push_back(right);
            return;
        }

        for (auto pos = ((1u << N) - 1u) & ~(left | mid | right); pos; pos &= pos - 1) {
            auto bit = pos & -pos;
            expand(
                N,
                M,
                depth + 1,
                (left | bit) << 1,
                mid | bit,
                (right | bit) >> 1);
        }
    }
};

std::uint64_t solve_parallel(int N, int M)
{
    NQueenExpand nqe;
    nqe.expand(N, M);
    const std::size_t length = nqe.left_ary.size();
    std::atomic<std::size_t> index(0);
    std::atomic<std::uint64_t> sum(0);
    std::vector<std::thread> vt;
    for (auto i = 0u; i < std::thread::hardware_concurrency(); ++i) {
        vt.emplace_back([&] {
        while (true) {
            std::size_t local_index = index++;
            if (local_index >= length) return;
            sum += solve(
                N,
                M,
                nqe.left_ary[local_index],
                nqe.mid_ary[local_index],
                nqe.right_ary[local_index]);
        }
        });
    }
    for (auto&& t : vt) t.join();
    return sum;
}

uint64_t solve_gpu(int N, int M)
{
    NQueenExpand nqe;
    nqe.expand(N, M);
    const std::size_t length = nqe.left_ary.size();
    std::uint32_t* left_ary_d;
    std::uint32_t* mid_ary_d;
    std::uint32_t* right_ary_d;
    REQUIRE(hipMalloc(
        (void**)&left_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    REQUIRE(hipMalloc(
        (void**)&mid_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    REQUIRE(hipMalloc(
        (void**)&right_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    std::uint64_t *result_d;
    REQUIRE(hipMalloc(
        (void**)&result_d, sizeof(std::uint64_t) * length) == hipSuccess);
    REQUIRE(hipMemcpy(
        left_ary_d,
        nqe.left_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        mid_ary_d,
        nqe.mid_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        right_ary_d,
        nqe.right_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    constexpr int threadsPerBlock = 16;
    const auto blockCount = (length + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(
        kernel,
        dim3(static_cast<uint32_t>(blockCount)),
        dim3(threadsPerBlock),
        0,
        nullptr,
        N,
        M,
        left_ary_d,
        mid_ary_d,
        right_ary_d,
        result_d,
        length);
    std::vector<std::uint64_t> result(length);
    REQUIRE(hipMemcpy(
        result.data(),
        result_d,
        sizeof(std::uint64_t) * length,
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(hipFree(left_ary_d) == hipSuccess);
    REQUIRE(hipFree(mid_ary_d) == hipSuccess);
    REQUIRE(hipFree(right_ary_d) == hipSuccess);
    REQUIRE(hipFree(result_d) == hipSuccess);

    return std::reduce(std::execution::par_unseq, cbegin(result), cend(result));
}

uint64_t solve_gpu_ver2(int N, int M)
{
    NQueenExpand nqe;
    nqe.expand(N, M);
    const std::size_t length = nqe.left_ary.size();
    std::uint32_t* left_ary_d;
    std::uint32_t* mid_ary_d;
    std::uint32_t* right_ary_d;
    REQUIRE(hipMalloc(
        (void**)&left_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    REQUIRE(hipMalloc(
        (void**)&mid_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    REQUIRE(hipMalloc(
        (void**)&right_ary_d, sizeof(std::uint32_t) * length) == hipSuccess);
    std::uint64_t* result_d;
    REQUIRE(hipMalloc(
        (void**)&result_d, sizeof(std::uint64_t) * length) == hipSuccess);
    REQUIRE(hipMemcpy(
        left_ary_d,
        nqe.left_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        mid_ary_d,
        nqe.mid_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        right_ary_d,
        nqe.right_ary.data(),
        sizeof(std::uint32_t) * length,
        hipMemcpyHostToDevice) == hipSuccess);
    constexpr int threadsPerBlock = 16;
    const auto blockCount = (length + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(
        kernel_ver2,
        dim3(static_cast<std::uint32_t>(blockCount)),
        dim3(threadsPerBlock),
        0,
        nullptr,
        N,
        M,
        left_ary_d,
        mid_ary_d,
        right_ary_d,
        result_d,
        length);
    std::vector<std::uint64_t> result(length);
    REQUIRE(hipMemcpy(
        result.data(),
        result_d,
        sizeof(std::uint64_t) * length,
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipFree(left_ary_d) == hipSuccess);
    REQUIRE(hipFree(mid_ary_d) == hipSuccess);
    REQUIRE(hipFree(right_ary_d) == hipSuccess);
    REQUIRE(hipFree(result_d) == hipSuccess);

    return std::reduce(std::execution::par_unseq, cbegin(result), cend(result));
}

TEST_CASE("N-Queens")
{
    const int N = 10;
    const int M = 5;

    BENCHMARK("CPU - Naive") { return solve(N); };
    BENCHMARK("CPU - Parallel") { return solve_parallel(N, M); };
    BENCHMARK("GPU - Parallel") { return solve_gpu(N, M); };
    BENCHMARK("GPU - Optimised") { return solve_gpu_ver2(N, M); };
}

__global__
void haxpy(unsigned int n, __half a, const __half* x, __half* y)
{
    const auto elts_per_lane{
        (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x)};
    const auto start = (threadIdx.x + blockDim.x * blockIdx.x) * elts_per_lane;

    for (auto i = 0u; i != std::min(elts_per_lane, n - start); ++i) {
        y[start + i] = __float2half(
            __half2float(a) *
            __half2float(x[start + i]) +
            __half2float(y[start + i]));
    }
}

__global__
void haxpy_native(unsigned int n, __half a, const __half* x, __half* y)
{
    const auto n2 = n/2;
    const __half2* x2 = (const __half2*)x;
    __half2* y2 = (__half2*)y;

    const auto elts_per_lane{
        (n2 + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x)};
    const auto start = (threadIdx.x + blockDim.x * blockIdx.x) * elts_per_lane;

    if (start > n2) return;

    for (auto i = 0u; i != std::min(elts_per_lane, n2 - start); ++i) {
        y2[start + i] =
            __hfma2(__halves2half2(a, a), x2[start + i], y2[start + i]);
    }
}

TEST_CASE("HAXPY")
{
    constexpr auto n{USHRT_MAX};
    const __half a{2.0f};

    __half* x;
    __half* y;

    REQUIRE(hipMalloc((void**)&x, n * sizeof(__half)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&y, n * sizeof(__half)) == hipSuccess);

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = half_float::half_cast<__half>(i);
    }

    constexpr auto blockSize{16};
    const auto nBlocks{std::thread::hardware_concurrency()};

    BENCHMARK("HAXPY") {
        hipLaunchKernelGGL(
            haxpy, dim3(nBlocks), dim3(blockSize), 0, nullptr, n, a, x, y);

        return hipDeviceSynchronize();
    };

    BENCHMARK("HAXPY-native") {
        hipLaunchKernelGGL(
            haxpy_native,
            dim3(nBlocks),
            dim3(blockSize),
            0,
            nullptr,
            n,
            a,
            x,
            y);

        return hipDeviceSynchronize();
    };
}