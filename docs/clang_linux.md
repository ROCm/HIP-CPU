# Building the HIP CPU Runtime and Client Applications CLANG C++ #

This tutorial shows how to use the Clang C++ Compiler on Linux to
build the HIP CPU Runtime. It does not delve into the details of HIP, the
C++ language, the CLANG toolset or the Linux ecosystem.

If you encounter any difficulties, [please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
for this tutorial.

## Prerequisites ##

To successfully complete this tutorial, the following steps are necessary:

1. Install [Intel TBB](https://github.com/oneapi-src/oneTBB):
   - At this time this is a pre-requisite for the Parallel Algorithms component
     of the C++ Standard Library implementations available in Linux;
   - In general it should be available from via your distro's package manager
     - Consider Ubuntu as an example:

       ```bash
       sudo apt-get install libtbb-dev
       ```

   - Alternatively, consider following the [build and install instructions for the latest version](https://github.com/oneapi-src/oneTBB/blob/master/cmake/README.md).
2. Install latest CMake
   - In general it should be available from via your distro's package manager
     - Consider Ubuntu as an example:

     ```bash
     sudo apt-get install cmake
     ```

   - Alternatively, [download and execute the installer](https://cmake.org/download/).
3. Install Git
   - [Follow the instructions](https://git-scm.com/download/linux) that apply
     to your environment.

## Ensure CLANG is Installed ##

To verify whether CLANG is installed and of a sufficiently recent version, open a
Terminal window and enter the following command:

```bash
clang -v
```

If it is not installed, you must install it
and the corresponding version of the [GNU C++ Library](https://clang.llvm.org/cxx_status.html)
using your distro's package manager. Consider Ubuntu 20.04 or newer as an
example:

```bash
sudo apt-get update
sudo apt-get install build-essential clang
```

## Clone and build the HIP CPU Runtime ##

```bash
git clone https://github.com/ROCm-Developer-Tools/HIP-CPU.git
cd HIP-CPU
mkdir build
cd build
cmake ../
cmake --build ./
```

## **OPTIONAL** Install the HIP CPU Runtime ##

```bash
# Assumes that you are in the build folder created in the build step.
cmake --build ./ --target install
```

## Verify the build by running the unit tests ##

```bash
# Assumes that you are in the build folder created in the build step.
ctest --output-on-failure
```

## Use the HIP CPU Runtime in your code ##

To use any of the HIP public interfaces include the `hip/hip_runtime.h` header.

- If you are working with CMake, link against the convenience `INTERFACE` target
  `hip_cpu_rt::hip_cpu_rt`, which is exported by the HIP CPU Runtime, which can
  be queried by `find_package(hip_cpu_rt)`;
- If you are not working with CMake, add either
  `/path_where_you_cloned_the_hip_cpu_runtime/include` or, if you installed it,
  `/path_where_you_installed_the_hip_cpu_runtime/include` to your include path.
