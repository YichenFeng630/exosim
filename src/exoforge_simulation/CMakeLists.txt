cmake_minimum_required(VERSION 3.28)
project(exoforge_simulation
    VERSION 1.0.0
    DESCRIPTION "A package for simulating biomechanical systems and exoskeletons."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set position independent code globally
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress developer warnings for dependencies
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
# set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

# Download CPM.cmake
file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
    EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Find packages
find_package(ament_cmake REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)

# Add GLFW via CPM
CPMAddPackage(
    NAME glfw
    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG 3.3.8
    OPTIONS
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_DOCS OFF"
    "GLFW_INSTALL OFF"
)

# Optional PNG support (not required by MuJoCo core)
option(EXOFORGE_USE_LODEPNG "Enable LodePNG for custom PNG IO (screenshots, texture loading)" OFF)

# Add LodePNG via CPM only if requested
if(EXOFORGE_USE_LODEPNG)
    CPMAddPackage(
        NAME lodepng
        GITHUB_REPOSITORY lvandeve/lodepng
        GIT_TAG master
        DOWNLOAD_ONLY YES
    )
    if(lodepng_ADDED)
        add_library(lodepng STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp)
        target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
        # Fix PIC issue for LodePNG
        set_target_properties(lodepng PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_options(lodepng PRIVATE -fPIC)
        message(STATUS "LodePNG enabled for exoforge_simulation.")
    endif()
else()
    message(STATUS "LodePNG disabled (EXOFORGE_USE_LODEPNG=OFF).")
endif()

# Add MuJoCo via CPM
CPMAddPackage(
    NAME mujoco
    GITHUB_REPOSITORY deepmind/mujoco
    GIT_TAG 3.3.5
    OPTIONS
    "MUJOCO_BUILD_EXAMPLES OFF"
    "MUJOCO_BUILD_SIMULATE ON"
    "MUJOCO_BUILD_TESTS OFF"
    "MUJOCO_TEST_PYTHON_UTIL OFF"
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exoforge_simulation
)

# Project source files
set(SOURCES
    # src/mj/simulate.cpp
)

# Create executable
# add_executable(mujoco_simulate ${SOURCES})

# Explicitly add include directories to the target
# target_include_directories(mujoco_imgui_simulate PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/exoforge_simulation
# )

# Link libraries
# target_link_libraries(mujoco_simulate
#     mujoco
#     lodepng
#     glfw
#     OpenGL::GL
#     ${CMAKE_DL_LIBS}
# )

# Platform-specific settings
# if(APPLE)
#     target_link_libraries(mujoco_simulate "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
#     set_target_properties(mujoco_simulate PROPERTIES
#         INSTALL_RPATH "@executable_path"
#         BUILD_WITH_INSTALL_RPATH TRUE
#     )
# elseif(UNIX)
#     target_link_libraries(mujoco_simulate pthread)
#     set_target_properties(mujoco_simulate PROPERTIES
#         INSTALL_RPATH "$ORIGIN"
#         BUILD_WITH_INSTALL_RPATH TRUE
#     )
# endif()

# Install the MuJoCo simulate executable
if(TARGET simulate)
    set_target_properties(simulate PROPERTIES OUTPUT_NAME mujoco_simulate)
    install(TARGETS simulate
        DESTINATION lib/${PROJECT_NAME}
    )
endif()

# Alternative: Install the simulate binary directly if it's built in a different location
# install(PROGRAMS ${mujoco_BINARY_DIR}/bin/simulate
#     DESTINATION lib/${PROJECT_NAME}
#     RENAME mujoco_simulate
# )

# Install targets
# install(TARGETS mujoco_simulate
#     DESTINATION lib/${PROJECT_NAME}
# )

# Install any additional files if needed (models, configs, etc.)
# install(DIRECTORY models/
#     DESTINATION share/${PROJECT_NAME}/models
# )

ament_package()
