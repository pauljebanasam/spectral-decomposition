# stn.split_tilde — standard min-devkit project script.
# This folder is designed to be dropped into  min-devkit/source/projects/
# where the devkit's top-level CMake auto-discovers it. min-posttarget
# converts the "_tilde" suffix to "~", producing the external stn.split~.
cmake_minimum_required(VERSION 3.19)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../min-api/script/min-pretarget.cmake)

include_directories("${C74_INCLUDES}")

set(SOURCE_FILES
    ${PROJECT_NAME}.cpp
)

add_library(${PROJECT_NAME} MODULE ${SOURCE_FILES})

# SDK-agnostic DSP core (header-only)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../min-api/script/min-posttarget.cmake)
