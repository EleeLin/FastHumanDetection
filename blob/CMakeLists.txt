include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SRCS "*.cpp")

#find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libraries ${OpenCV_LIBS}")

add_library(blob ${SRCS})
target_link_libraries(blob ${OpenCV_LIBS})
