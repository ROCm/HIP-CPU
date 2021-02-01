set(name bit_extract)

add_executable(${name} bit_extract.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)