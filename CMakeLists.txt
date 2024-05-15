
cmake_minimum_required(VERSION 3.10)
project(matplotlibcpp)

set(DEFAULT_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0 -g -ggdb -w")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include(third_party.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/src/)

add_library(optsolver SHARED 
                src/unconstraintsolver.cc 
                src/constraintsolver.cc 
                src/common.cc)

target_link_libraries(optsolver ${third_party_libs})


add_executable(example src/example.cc)
target_link_libraries(example optsolver ${third_party_libs})

add_executable(example1 src/example1.cc)
target_link_libraries(example1 optsolver ${third_party_libs})

add_executable(example2 src/example2.cc)
target_link_libraries(example2 optsolver ${third_party_libs})

add_executable(example3 src/example3.cc)
target_link_libraries(example3 optsolver ${third_party_libs})





