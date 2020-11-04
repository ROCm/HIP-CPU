# How to Contribute Changes #

## Contribution Steps ##

* [Build and debug the artifacts and / or user application](docs/building.md)
* File an [issue][https://github.com/ROCm-Developer-Tools/HIP-CPU/issues] and
  create a [pull request](https://github.com/ROCm-Developer-Tools/HIP-CPU/pulls)
  with the change and we will review it.
* If the change impacts existent functionality, or adds new functionality, add a
  line describing the change to [**CHANGELOG.md**](CHANGELOG.md).
* Please add a unit test in the corresponding `.cpp` file under `/src`; see
  [The Pitchfork Layout](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.tests.merged)
  convention for merged test placement.
* If the change adds functionality that is covered by an extant test that is
  part of the [HIP GPU Runtime](https://github.com/ROCm-Developer-Tools/HIP),
  please consider adding it in a new `.cpp` file under under `/tests`.
* Run tests:
  * Via CTest;
  * By executing the relevant build artifact:
    * `legacy_tests` for tests ported from the [HIP GPU Runtime](https://github.com/ROCm-Developer-Tools/HIP);
    * `public_interface_tests` for unit tests;
    * `performance_tests` for benchmarks.

## About the Code ##

* [The Pitchfork Layout](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs)
  is used to physically organise files; the [separate header](https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.header-placement.separate)
  convention is adopted.
* The [PPP Style Guide](https://stroustrup.com/Programming/PPP-style.pdf)
  governs coding style; 80-character line-length is mandatory.
* The HIP CPU Runtime is fully implemented in C++, and a compliant C++
  development environment is required
  * C++17 is the earliest supported language standard;
  * [Full support](https://en.cppreference.com/w/cpp/compiler_support#cpp17) is
    required both in the compiler and in the standard library; the HIP CPU
    Runtime will try to signal at compile time if features are missing.
