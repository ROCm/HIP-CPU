# The HIP CPU Runtime #

The HIP CPU Runtime enables cross-platform [HIP](https://github.com/ROCm-Developer-Tools/HIP)
development on multiple operating systems such as Linux, macOS on Windows, when
targeting CPUs. It is a header-only C++ library that is not operating system or
target specific. It does not require special compiler support, and allows the
same HIP code to be written once to run (almost) everywhere.

## System Requirements ##

- A C++17 (or later) compliant toolchain, e.g.:
  - [GCC](https://gcc.gnu.org/), from version 9 onwards;
  - [Clang / LLVM](http://clang.llvm.org/), from version 9 onwards;
  - [Microsoft C++](https://visualstudio.microsoft.com/vs/features/cplusplus/),
    from version 19.14 onwards;
- Please check [cppreference.com](https://en.cppreference.com/w/cpp/compiler_support)
  for more details about C++ support on various toolchains.

## Introduction ##

Consider the following HIP code:

```cpp
#include <hip/hip_runtime.h>

__global__
void kernel(const int* pA, const int* pB, int* pC) {
    const auto gidx = blockIdx.x * blockDim.x + threadIdx.x;

    pC[gidx] = pA[gidx] + pB[gidx];
}

int main() {
    int a[]{1, 2, 3, 4, 5};
    int b[]{6, 7, 8, 9, 10};
    int c[sizeof(a) / sizeof(a[0])];

    int* pA{nullptr}; hipMalloc((void**)&pA, sizeof(a));
    int* pB{nullptr}; hipMalloc((void**)&pB, sizeof(b));
    int* pC{nullptr}; hipMalloc((void**)&pC, sizeof(c));

    hipMemcpy(pA, a, sizeof(a));
    hipMemcpy(pB, b, sizeof(b));

    hipLaunchKernelGGL(
        kernel,
        dim3(1),
        dim3(sizeof(a) / sizeof(a[0])),
        0,
        nullptr,
        pA,
        pB,
        pC);

    hipMemcpy(c, pC, sizeof(c));

    for (auto i = 0u; i != sizeof(a) / sizeof(a[0]); ++i) {
      if (c[i] != a[i] + b[i]) throw;
    }

    return 0;
}
```

Without the HIP CPU Runtime, this would only execute on some AMD or NVIDIA GPUs,
with various requirements being imposed on the surrounding environment (e.g.
only the Linux OS can be used in AMD's case, only certain compilers can be used,
only IHV provided tools work etc.). Using the HIP CPU Runtime, the same code can
be compiled and executed, unmodified, on any CPU, in any OS. All familiar tools
such as debuggers, sanitizers, profilers or runtime optimisers can be used
without any extra ritual involved. Furthermore, because the HIP programming
model is centered on the idea of bulk, blocked parallel execution, it implicitly
"forces" the programmer to adhere to a number of best practices that lead to
better performance than that which obtains via e.g. haphazard parallelism via
`std::thread`s. Finally, the vast array of CUDA/HIP libraries that have hitherto
been unusable without a GPU, become accessible to the wider programmer
population.

## Portability ##

Code using the portable subset of HIP will execute correctly, without
modification, across CPU and GPU targets, be it when the HIP CPU Runtime or the
HIP GPU Runtime is used. Using features that are part of the non-portable subset
prevents such portability, and in some cases will not work at all. Examples of
inherently non-portable features are:

- inline PTX / inline GCN ASM
- reliance of lock-stepped execution of fibers in a warp
  - technically this is UB even in current HIP / CUDA

## Completeness ##

The HIP CPU Runtime is not a complete implementation of HIP, at the moment. We
do intend to achieve full-convergence. If you identify a piece of functionality
that you need and is not yet implemented, [please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
, preferably including a minimal use-case / example of said feature's use.

## Performance ##

The HIP CPU Runtime is implemented via the [Parallel Algorithms](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0024r2.html)
component of the C++ Standard Library. Thus, its performance is conditioned by
the latter's Quality of Implementation. At the same, the compiler's ability to
auto-vectorise is another determinant of achievable performance, as the HIP CPU
Runtime does not do any explicit vectorisation and does not rely on a
vectorising back-end. This choice is motivated by the desire to maximise
portability and to avoid relying on any non-standard functionality. Given the
above, it follows that relatively significant variations in the performance
achieved when using the HIP CPU Runtime can be observed as both compilers and
C++ Standard Library implementations evolve.

## Feedback ##

When encountering issues or coming up with suggestions for the HIP CPU Runtime,
please file [issues and suggestions](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose).
If you want to join us and contribute to the development effort please have a
peek at [the guidelines](/CONTRIBUTING.md) and then cry havoc and
[let loose the Pull Requests](https://github.com/ROCm-Developer-Tools/HIP-CPU/pulls).