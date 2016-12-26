cmake_minimum_required(VERSION 2.8)
project(ImageClassifier)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)



add_executable(ImageClassifier svm/Source.cpp svm/Utilities.h svm/Utilities.cpp)
target_link_libraries(ImageClassifier ${OpenCV_LIBS} lopencv_nonfree)