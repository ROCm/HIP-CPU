set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(public_interface_tests
    catch_main.cpp
    hip_api.test.cpp
    hip_atomic.test.cpp
    hip_constants.test.cpp
    hip_defines.test.cpp
    hip_device_launch_parameters.test.cpp
    hip_enums.test.cpp
    hip_fp16.test.cpp
    hip_math.test.cpp
    hip_runtime_api.test.cpp
    hip_runtime.test.cpp
    hip_types.test.cpp
    hip_vector_types.test.cpp
)
target_link_libraries(public_interface_tests
    PRIVATE examples_common
)

add_test(
    NAME public_interface_tests
    COMMAND public_interface_tests
)