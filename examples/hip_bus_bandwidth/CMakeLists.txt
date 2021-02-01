set(name hip_bus_bandwidth)

add_executable(${name} hip_bus_bandwidth.cpp ResultDatabase.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)