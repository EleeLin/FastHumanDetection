include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SRCS "*.cpp")
file(GLOB HDRS "*.h")

add_library(background ${SRCS} ${HDRS})
target_link_libraries(background ${OpenCV_LIBS} blob amf)

add_library(bgsub SHARED ${SRCS} ${HDRS})
target_link_libraries(bgsub ${OpenCV_LIBS} background amf)

add_dependencies(bgsub background blob amf)
