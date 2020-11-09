set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

include(CTest)
enable_testing()

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
        hip_host_module.cpp
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

add_test(NAME legacy_tests COMMAND legacy_tests)