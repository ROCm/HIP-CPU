set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

include(CTest)
enable_testing()

if (${CMAKE_HOST_APPLE}) # Add brew-ed TBB.
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
endif()

add_executable(performance_tests catch_benchmarking_main.cpp benchmarks.cpp)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    target_compile_options(
        performance_tests PRIVATE
            /arch:AVX2
            /fp:fast
            /openmp:experimental
            /permissive-
            /W4)
    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL Debug
        AND NOT ${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
        target_compile_options(performance_tests PRIVATE /Ob3)
    endif ()
else ()
    target_compile_options(
        performance_tests PRIVATE -ffast-math -fopenmp-simd -march=native -Wall)
    target_link_libraries(performance_tests PRIVATE dl tbb)
endif ()
target_include_directories(
    performance_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(performance_tests PRIVATE Threads::Threads)

add_executable(
    legacy_tests
        catch_main.cpp
        hip_atomics.cpp
        hip_device_ballot.cpp
        hip_device_clock.cpp
        hip_device_clz.cpp
        hip_device_dynamic_shared.cpp
        hip_device_dynamic_shared_2.cpp
        hip_device_ffs.cpp
        hip_device_fma.cpp
        hip_device_get_limit.cpp
        hip_device_half_half2.cpp
        hip_device_malloc.cpp
        hip_device_math_double.cpp
        hip_device_math_float.cpp
        hip_device_memset.cpp
        hip_device_popc.cpp
        hip_device_printf.cpp
        hip_device_std_complex.cpp
        hip_device_synchronize.cpp
        hip_get_device.cpp
        hip_host_async_memcpy_all.cpp
        hip_host_hipevent_elapsed_time.cpp
        hip_host_hipevent_event_record.cpp
        hip_host_math_float_double.cpp
        hip_host_memcpy_all.cpp
        # hip_host_module.cpp
        hip_host_multithread_hipstream_memcpy.cpp
        hip_host_multithread_hipstream.cpp
        hip_host_multithread_memcpy.cpp
        hip_set_device_flags.cpp
        hip_set_device.cpp
        hip_vector_types.cpp)
if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    target_compile_options(legacy_tests PRIVATE -fopenmp-simd)
    target_link_libraries(legacy_tests PRIVATE dl tbb)
endif ()
target_include_directories(
    legacy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(legacy_tests PRIVATE Threads::Threads)

add_test(NAME "legacy_atomic" COMMAND legacy_tests [device][atomic])
add_test(NAME "legacy_ballot" COMMAND legacy_tests [device][ballot])
add_test(NAME "legacy_clock" COMMAND legacy_tests [device][clock])
add_test(NAME "legacy_clz" COMMAND legacy_tests [device][clz])
add_test(
    NAME "legacy_Dynamic_SharedMem_I"
    COMMAND legacy_tests [device][dynamic_shared_1])
add_test(
    NAME "legacy_Dynamic_SharedMem_II"
    COMMAND legacy_tests [device][dynamic_shared_2])
add_test(NAME "legacy_ffs" COMMAND legacy_tests [device][ffs])
add_test(NAME "legacy_fma_float" COMMAND legacy_tests [device][fmaf])
add_test(NAME "legacy_fma___half" COMMAND legacy_tests [device][fmah])
add_test(
    NAME "legacy_hipDeviceGetLimit"
    COMMAND legacy_tests [host][hipDevice][get_limit])
add_test(NAME "legacy___half_&&___half2" COMMAND legacy_tests [device][half])
add_test(
    NAME "legacy_hipDeviceMalloc" COMMAND legacy_tests [device][hipMalloc])
add_test(NAME "legacy_double_math" COMMAND legacy_tests [device][math][double])
add_test(NAME "legacy_float_math" COMMAND legacy_tests [device][math][float])
add_test(NAME "legacy_half_math" COMMAND legacy_tests [device][math][half])
add_test(
    NAME "legacy_hipDeviceMemset" COMMAND legacy_tests [device][hipMemset])
add_test(NAME "legacy_popc" COMMAND legacy_tests [device][popc])
add_test(NAME "legacy_printf" COMMAND legacy_tests [device][printf])
add_test(
    NAME "legacy_std_complex" COMMAND legacy_tests [device][std::complex])
add_test(
    NAME "legacy_hipDeviceSynchronize"
    COMMAND legacy_tests [host][hipDevice])
add_test(
    NAME "legacy_hipGetDevice"
    COMMAND legacy_tests [host][hipDevice][get_device])
add_test(NAME "legacy_async_memcpy" COMMAND legacy_tests [host][memcpy_async])
add_test(NAME "legacy_hipEvent_t" COMMAND legacy_tests [host][hipEvent_t])
# add_test(NAME "legacy_module_API" COMMAND legacy_tests [host][module])
add_test(NAME "legacy_hipStream_t" COMMAND legacy_tests [host][hipStream_t])
add_test(NAME "legacy_host_math" COMMAND legacy_tests [host][math])
add_test(NAME "legacy_memcpy" COMMAND legacy_tests [host][memcpy])
add_test(
    NAME "legacy_hipSetDeviceFlags"
    COMMAND legacy_tests [host][hipDevice][set_device_flags])
add_test(
    NAME "legacy_hipSetDevice"
    COMMAND legacy_tests [host][hipDevice][set_device])
add_test(NAME "legacy_vector_types" COMMAND legacy_tests [host][vector_types])