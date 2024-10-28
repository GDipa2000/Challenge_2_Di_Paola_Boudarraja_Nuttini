cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ChallengeTwo VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(challenge_two Challenge_two.cpp)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories for stb_image and stb_image_write
include_directories(include)

# Link Eigen3
target_link_libraries(challenge_two Eigen3::Eigen)

# Copy the necessary files to the build directory
add_custom_command(
    TARGET challenge_two POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/Einstein.jpg
            ${CMAKE_BINARY_DIR}/Einstein.jpg
)
