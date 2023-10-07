project(ncnnyolox)

cmake_minimum_required(VERSION 3.14.7)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-3.4.18-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230517-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnnSAM SHARED samncnn.cpp segment_anything.cpp sam.cpp ndkcamera.cpp)

target_link_libraries(ncnnSAM ncnn ${OpenCV_LIBS} camera2ndk mediandk)
