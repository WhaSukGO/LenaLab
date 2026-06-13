cmake_minimum_required(VERSION 3.16)
project(vio_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(vio main.cpp)
target_link_libraries(vio PRIVATE Ceres::ceres ${OpenCV_LIBS} Eigen3::Eigen)
