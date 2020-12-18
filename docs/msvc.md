# Building the HIP CPU Runtime and Client Applications with Microsoft C++ #

This tutorial shows how to use the Microsoft Visual C++ Compiler on Windows to
build the HIP CPU Runtime. It does not delve into the details of HIP, the
C++ language or the Microsoft C++ toolset.

If you encounter any difficulties, [please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
for this tutorial.

## Prerequisites ##

To successfully complete this tutorial, the following steps are necessary:

1. Install the Microsoft Visual C++ (MSVC) compiler toolset:
   - If you have a recent version of Visual Studio, launch the Visual Studio
     Installer and ensure that the C++ workload is enabled. If it is not, check
     the relevant box and select **Modify** in the installer;
   - Alternatively, you can install only the **C++ Build Tools** without a full
     Visual Studio IDE installation. From the Visual Studio [Downloads](https://visualstudio.microsoft.com/downloads#other)
     page, select the download for **Build Tools for Visual Studio** under the
     **All Downloads**/**Tools for Visual Studio** entry, and execute the
     downloaded file;
     - This shall launch the Visual Studio Installer; under the available
       Visual Studio Build Tools workloads pick **C++ Build Tools** and select
       **Install**;
     - Note that at the moment it appears that the above flow for downloading
       the installer is broken, therefore please consider using the
       [direct download](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16#)
2. Install latest CMake
   - [Download and execute the installer](https://cmake.org/download/)
3. Install Git
   - [Download and execute the installer](https://git-scm.com/download/win)

## Clone and build the HIP CPU Runtime ##

```cmd
git clone https://github.com/ROCm-Developer-Tools/HIP-CPU.git
cd HIP-CPU
mkdir build
cd build
cmake ../
cmake --build ./
```

## **OPTIONAL** Install the HIP CPU Runtime ##

```cmd
rem Assumes that you are in the build folder created in the build step.
cmake --build ./ --target install
```

## Verify the build by running the unit tests ##

```cmd
rem Assumes that you are in the build folder created in the build step.
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
- For potentially enhanced vectorisation, consider adding `/openmp:experimental`
  to your compiler flags, as this will allow the back-end to (eventually) be
  more aggressive in vectorising HIP code.
