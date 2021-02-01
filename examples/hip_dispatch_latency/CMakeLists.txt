set(name hip_dispatch_latency)

add_executable(${name} hipDispatchLatency.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)