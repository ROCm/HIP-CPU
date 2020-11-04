find_package(Threads REQUIRED)

add_executable(
    public_interface_tests
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
        hip_vector_types.test.cpp)
if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    target_compile_options(public_interface_tests PRIVATE -fopenmp-simd)
    target_link_libraries(public_interface_tests PRIVATE dl tbb)
else ()
    target_compile_definitions(
        public_interface_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif ()
target_include_directories(
    public_interface_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../external)
target_link_libraries(public_interface_tests PRIVATE Threads::Threads)

add_test(NAME public_interface_tests COMMAND public_interface_tests)