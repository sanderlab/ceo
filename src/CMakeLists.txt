cmake_minimum_required(VERSION 2.8.12)
project(ceo-clustering)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

add_executable(ceo-clustering main.cpp ceo.cpp)
target_link_libraries(ceo-clustering ${CONAN_LIBS})

add_custom_command(TARGET ceo-clustering POST_BUILD
COMMAND echo built with the flags: ${CMAKE_CXX_FLAGS})
