cmake_minimum_required(VERSION 3.22.1)
project(bandbuddy_audio)

add_library(bandbuddy_audio SHARED native_demucs.cpp)
target_compile_options(bandbuddy_audio PRIVATE -O3)

find_library(log-lib log)
target_link_libraries(bandbuddy_audio ${log-lib})
