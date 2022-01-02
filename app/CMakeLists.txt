cmake_minimum_required(VERSION 3.4.1)

include_directories(${PROJECT_SOURCE_DIR}/src/main/cpp/arm)
include_directories(../OpenCVLib430/native/jni/include)

add_library(armnn SHARED IMPORTED)
set_target_properties(armnn PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libarmnn.so)
add_library(armnnUtils STATIC IMPORTED)
set_target_properties(armnnUtils PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libarmnnUtils.a)
add_library(armnnTfLiteParser SHARED IMPORTED)
set_target_properties(armnnTfLiteParser PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libarmnnTfLiteParser.so)
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)

add_library(nnbackend SHARED src/main/cpp/nnbackend.cpp)
find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

target_link_libraries(nnbackend armnn armnnUtils armnnTfLiteParser lib_opencv jnigraphics ${log-lib})