cmake_minimum_required(VERSION 3.15)
project(music_dance_motion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# DynamixelSDK (프로젝트 third_party 에 이미 빌드되어 있음: libdxl_sbc_cpp.so)
set(DXL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/DynamixelSDK-3.7.31)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
  set(DXL_BUILD linux_sbc)
  set(DXL_LIB dxl_sbc_cpp)
else()
  set(DXL_BUILD linux64)
  set(DXL_LIB dxl_x64_cpp)
endif()

# libsndfile (WAV 읽기) + ALSA (재생)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(ALSA REQUIRED alsa)

add_executable(dance main.cpp)

target_include_directories(dance PRIVATE
  ${DXL_DIR}/c++/include/dynamixel_sdk
  ${SNDFILE_INCLUDE_DIRS}
  ${ALSA_INCLUDE_DIRS}
)
target_link_directories(dance PRIVATE
  ${DXL_DIR}/c++/build/${DXL_BUILD}
)
target_link_libraries(dance PRIVATE
  ${DXL_LIB}
  ${SNDFILE_LIBRARIES}
  ${ALSA_LIBRARIES}
  pthread
)

# 런타임에 libdxl_sbc_cpp.so 를 찾도록 rpath 지정
set_target_properties(dance PROPERTIES
  BUILD_RPATH ${DXL_DIR}/c++/build/${DXL_BUILD}
)
