set(name square)

add_executable(${name} square.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)