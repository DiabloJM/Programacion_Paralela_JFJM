cmake_minimum_required(VERSION 3.26)
project(HolaImagen)

set(CMAKE_CXX_STANDARD 17)

add_executable(HolaImagen main.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(HolaImagen ${OpenCV_LIBS})