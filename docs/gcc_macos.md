# Building the HIP CPU Runtime and Client Applications GCC C++ #

This tutorial shows how to use the GCC C++ Compiler on MacOS to
build the HIP CPU Runtime. It does not delve into the details of HIP, the
C++ language, the GCC toolset or the MacOS ecosystem.

If you encounter any difficulties, [please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
for this tutorial.

## Prerequisites ##

To successfully complete this tutorial, the following steps are necessary:

1. Install [Homebrew](https://brew.sh):
   - LLVM at the moment does not have parallel algorithm support. However,
     it is available in GNU G++. Homebrew provides a straightforward package
	 manager for MacOS, which allows both the GNU G++ compiler, and Intel
	 TBB on which HIP-CPU depends. One can also consider other package managers
	 such as MacPorts, but we will focus on Homebrew here.
	 
   - go to [The Homebrew Site](https://brew.sh) and follow the installation 
     instructions there:
	 ```bash
	 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	 ```
	 You may or may not get messages about needing to install the XCode command line
	 utilities if these are not yet installed. Follow the appropriate installation
	 instructions on the Homebrew site.
	 
2. Install [GNU G++][(https://gcc.gnu.org)
	 - Using Homebrew:
	   ```bash
	   brew install gcc@10
	   ```
   - to avoid confusion with the system GCC (which is actually Clang), homebrew will name the
     executables as gcc-10 and g++-10
	 
3. Install [Intel TBB](https://github.com/oneapi-src/oneTBB):
   - At this time this is a pre-requisite for CPU-HIP (makefiles depend on it)
     as it used to be a prerequisite for parallel algorithms for G++ on Linux.
	 Under MacOS, with the Homebrew installation, I successfully compiled a simple
	 parallel C++-17 parallel algorithm test from [here](https://solarianprogrammer.com/2019/05/09/cpp-17-stl-parallel-algorithms-gcc-intel-tbb-linux-macos/)
	 without TBB. However it is very easy to install using Homebrew:
	 ```bash
	 brew install tbb
	 ```
2. Install latest CMake
   - Homebrew makes this super easy also:
     ```bash
     brew install cmake
     ```

   - Alternatively, [download and execute the installer](https://cmake.org/download/).
3. Install Git
   - This can be done from Homebrew:
     ```bash
     brew install git
	 ```
   - alternatively a Git client can be installed from [The Git Website](https://git-scm.com)`
  
Ensure that `/usr/local/bin` is on your PATH -- this is where Homebrew installs all its tools.
 
## Ensure GCC is Installed ##

To verify whether GCC is installed and of a sufficiently recent version, open a
Terminal window and enter the following command:

```bash
gcc-10 -v
```

## Clone and build the HIP CPU Runtime ##

```bash
git clone https://github.com/ROCm-Developer-Tools/HIP-CPU.git
cd HIP-CPU
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-10 ,,/
cmake --build ./
```

If you want to later install the HIP CPU Runtime in non standard folder you can
specify the install location to the `cmake` command 

```bash
cmake -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_INSTALL_PREFIX=<location to install the runtime> 
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
