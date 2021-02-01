set(name streams)

add_executable(${name} streams.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)