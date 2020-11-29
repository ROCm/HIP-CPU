# HIP CPU Runtime Performance Considerations #

## Recommended Reading ##

- [C++ Core Guidelines - Performance](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-performance)
- [CPU Caches and Why You Care](https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)
- [Optimizing Software in C++](https://www.agner.org/optimize/optimizing_cpp.pdf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)

## General Considerations ##

1. Prefer larger block sizes when not using barriers
2. Prefer doing more work per block element (HIP / CUDA thread)
   - Avoid bijective mappings where each element in the execution domain
     performs only one computation on a single element from the data domain, as
     these do not amortise setup / tear-down costs;
   - See the following for an argument in favour of this being desirable on the
     GPU as well:
     - [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)
     - [Unrolling Parallel Loops](https://www.nvidia.com/docs/IO/116711/sc11-unrolling-parallel-loops.pdf)
     - [Use Registers and Multiple Outputs per Thread on GPU](http://laurel.datsi.fi.upm.es/_media/proyectos/gopac/volkov10-pmaa.pdf)
3. **TODO**

## Pitfalls ##

1. Barriers
   - Due to their reliance on `O(block_size)` fiber switches, functions that
     have barrier semantics, such as `__syncthreads()`, induce significant
     slowdown;
   - It is preferable to avoid using barriers if possible, especially since,
     unlike on GPUs, `__shared__` memory does not provide performance benefits;
   - If you must use barriers, prefer smaller block sizes such as `8` or `16`.
2. `FP16` i.e. `__half`
   - The HIP CPU Runtime provides correct but low-performance support for `FP16`
     computation, in order to ensure that code which uses `__half` or `__half2`
     is portable;
   - It is preferable to avoid using `__half` or `__half2` arithmetic.
3. Passing large arguments by-value to `__global__` functions
   - This is a commonly encountered anti-pattern, stemming from the presence of
     dedicated memory for arguments on some (generally older) GPUs;
     - It is generally disadvantageous when targeting an AMD GPUs, thus the
       guidance below applies to them as well, and leads to performance portable
       code;
   - If `sizeof(T) > 32` for a type `T` that is the type of an argument passed
     to a function, strongly prefer pass-by-pointer / pass-by-reference.
4. Excessive unrolling via `#pragma unroll`
   - This is a commonly encountered anti-pattern, stemming from historical
     weaknesses in GPU compiler optimisation, which are no longer present in
     modern toolchains;
   - Can be extremely harmful due to trashing the I$
     - See [mixbench](https://github.com/ekondis/mixbench/blob/bab3fb724572547015da58d4d90f263e4a0552cd/mixbench-hip/mix_kernels_hip.cpp#L61)
     for an example;
   - Strongly prefer deferring to the compiler on matters of unrolling.
5. Composition with MPI
   - The interaction between the underlying implementation of the C++ Standard
     Library and, more specifically, its Parallel Algorithms component, can and
     will interact in opaque ways with any MPI driven scheduling;
   - Experiment with pinning / affinity of MPI tasks if performance is low in
     such cases.
