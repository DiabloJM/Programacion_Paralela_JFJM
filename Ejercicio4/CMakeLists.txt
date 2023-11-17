cmake_minimum_required(VERSION 3.26)
project(Ejercicio4 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Ejercicio4 main.cu)

set_target_properties(Ejercicio4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
