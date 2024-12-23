cmake_minimum_required(VERSION 3.10)
project(element_yolov5)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs -g")

include_directories("${SOPHON_SDK_SOC}/include/")
include_directories("${SOPHON_SDK_SOC}/include/opencv4")
link_directories("${SOPHON_SDK_SOC}/lib/")
set(BM_LIBS bmlib bmrt bmcv yuv)
find_library(BMJPU bmjpuapi)
if(BMJPU)
    set(JPU_LIBS bmjpuapi bmjpulite)
endif()

# include_directories(../)
# include_directories(../../../framework)
# include_directories(../../../framework/include)

# include_directories(../../../3rdparty/spdlog/include)
# include_directories(../../../3rdparty/nlohmann-json/include)
# include_directories(../../../3rdparty/httplib)

include_directories(include)
add_library(${PROJECT_NAME} SHARED
    src/yolov5_pre_process.cc
    src/yolov5_post_process.cc
    src/yolov5_inference.cc
    src/yolov5.cc
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${FFMPEG_LIBS} ${OpenCV_LIBS} ${BM_LIBS} ${JPU_LIBS} stream -fprofile-arcs -lgcov -lpthread)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/httplib)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/nvr_backend/stream_elements/algorithm/)