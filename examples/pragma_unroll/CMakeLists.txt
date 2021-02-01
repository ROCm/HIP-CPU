set(name pragma_unroll)

add_executable(${name} pragma_unroll.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)