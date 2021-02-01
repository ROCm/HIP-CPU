set(name vadd_hip)

add_executable(${name} vadd_hip.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)