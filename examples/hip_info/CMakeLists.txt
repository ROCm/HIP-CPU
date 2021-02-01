set(name hip_info)

add_executable(${name} hipInfo.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)