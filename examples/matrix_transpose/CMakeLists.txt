set(name matrix_transpose)

add_executable(${name} MatrixTranspose.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)