set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

if (${CMAKE_HOST_APPLE}) # Add brew-ed TBB.
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
endif()

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