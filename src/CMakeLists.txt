set(CMAKE_CXX_EXTENSIONS OFF)

add_library(src_common INTERFACE)

target_link_libraries(src_common
    INTERFACE
        hip_cpu_rt::hip_cpu_rt
)
target_compile_options(src_common
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
target_compile_definitions(src_common
    INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        >
)

if (${CMAKE_HOST_APPLE}) # Add brew-ed TBB.
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
endif()

add_subdirectory(hip)