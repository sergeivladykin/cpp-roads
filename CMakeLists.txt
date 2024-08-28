# OpenCV installing: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
# or sudo apt install libopencv-dev ???

# Torch can be get (prebuilt libraries) from here
# https://pytorch.org/get-started/locally/
# Select
#Stable (2.3.1)
#Linux
#LibTorch
#C++ / Java
#CPU
#https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcpu.zip
#https://pytorch.org/cppdocs/installing.html
#https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5
             
# xtensor is header libra2ry that requires xtl
# sudo apt install doesn't work!
# https://github.com/xtensor-stack/xtl.git
# mkdir -p build && cd build
# cmake -S . -D CMAKE_INSTALL_PREFIX=/home/svladykin/AIR/cpp-deps/xtl
# make install
# https://github.com/xtensor-stack/xtensor
# mkdir -p build && cd build
# cmake -S .. -DCMAKE_INSTALL_PREFIX=/home/svladykin/AIR/xtensor -DCMAKE_PREFIX_PATH=/home/svladykin/AIR/xtl
# make install

# sudo apt-get install libcanberra-gtk-module
# sudo apt-get install libgtk2.0-dev



cmake_minimum_required(VERSION 2.18 FATAL_ERROR)
project(cpp-roads)
set(CMAKE_CXX_STANDARD 17)


add_executable( cpp-roads main.cpp )

target_compile_features(cpp-roads PUBLIC cxx_std_17)

# opencv - installed in /usr/local
find_package( OpenCV REQUIRED )
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

#libtorch - downloaded prebuilt
set (CMAKE_PREFIX_PATH "deps/libtorch")
find_package( Torch  REQUIRED )
message(STATUS "Torch_DIR = ${Torch_DIR}")
message(STATUS "TORCH_INSTALL_PREFIX = ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
#find_library(opencv_imgcodecs)

# xtensor - headers only lib
set (CMAKE_PREFIX_PATH "deps/xtl")
find_package( xtl REQUIRED )
message (STATUS "xtl_INCLUDE_DIRS = ${xtl_INCLUDE_DIRS}")

set (CMAKE_PREFIX_PATH "deps/xtensor")
find_package( xtensor REQUIRED )
message (STATUS "xtensor_INCLUDE_DIRS = ${xtensor_INCLUDE_DIRS}")

target_include_directories( cpp-roads PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_include_directories( cpp-roads PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories( cpp-roads PRIVATE ${xtensor_INCLUDE_DIRS})

# matplotlib requires python and numpy
find_package (matplotlib_cpp)
target_include_directories( cpp-roads PRIVATE matplotlib_cpp_INCLUDE_DIRS)
message (STATUS "matplotlib_cpp_INCLUDE_DIRS = ${matplotlib_cpp_INCLUDE_DIRS}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories( cpp-roads PRIVATE ${Python3_INCLUDE_DIRS} )
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")

find_package(Python3 COMPONENTS NumPy REQUIRED)
target_include_directories( cpp-roads PRIVATE ${Python3_NumPy_INCLUDE_DIRS} )
message(STATUS "Python3_NumPy_INCLUDE_DIRS = ${Python3_NumPy_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD_LIBRARIES -lopencv_imgcodecs)
target_link_libraries( cpp-roads ${OpenCV_LIBS} )
target_link_libraries( cpp-roads ${TORCH_LIBRARIES})
target_link_libraries( cpp-roads xtensor)
# for matplotlib
target_link_libraries( cpp-roads ${Python3_LIBRARIES})
