set(name shared_memory)

add_executable(${name} sharedMemory.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)