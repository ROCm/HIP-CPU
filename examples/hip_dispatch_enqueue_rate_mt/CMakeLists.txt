set(name hip_dispatch_enqueue_rate_mt)

add_executable(${name} hipDispatchEnqueueRateMT.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)