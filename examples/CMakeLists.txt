set(CMAKE_CXX_EXTENSIONS OFF)

add_library(examples_common INTERFACE)

target_link_libraries(examples_common
    INTERFACE
        hip_cpu_rt::hip_cpu_rt
)
target_compile_options(examples_common
    INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            /arch:AVX2
            /permissive-
            /Qvec-report:2
            /Zc:preprocessor
        >
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
            -Wall
        >
)
target_compile_definitions(examples_common
    INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        >
)

add_subdirectory(bit_extract)
add_subdirectory(hip_bus_bandwidth)
add_subdirectory(hip_dispatch_enqueue_rate_mt)
add_subdirectory(hip_dispatch_latency)
add_subdirectory(hip_event)
add_subdirectory(hip_info)
add_subdirectory(matrix_transpose)
add_subdirectory(occupancy)
add_subdirectory(pragma_unroll)
add_subdirectory(shared_memory)
add_subdirectory(square)
add_subdirectory(streams)
add_subdirectory(vadd_hip)