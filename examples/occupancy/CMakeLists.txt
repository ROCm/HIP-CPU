set(name occupancy)

add_executable(${name} occupancy.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)