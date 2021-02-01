set(name hip_event)

add_executable(${name} hipEvent.cpp)

target_link_libraries(${name}
    PRIVATE examples_common
)