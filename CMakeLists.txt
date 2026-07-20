cmake_minimum_required(VERSION 3.15)
project(mlrisk VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # Add /WX after stabilization
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    # Add -Werror after stabilization
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
endif()

# Library sources
set(MLRISK_SOURCES
    src/rolling.c
    src/sizing.c
    src/vol.c
    src/split.c
    src/linreg.c
)

# Public headers
set(MLRISK_HEADERS
    include/mlrisk/types.h
    include/mlrisk/rolling.h
    include/mlrisk/sizing.h
    include/mlrisk/vol.h
    include/mlrisk/split.h
    include/mlrisk/linreg.h
    include/mlrisk/mlrisk.h
)

# Create static library
add_library(mlrisk STATIC ${MLRISK_SOURCES} ${MLRISK_HEADERS})

target_include_directories(mlrisk PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# libm is a separate library on Unix toolchains
if(UNIX)
    target_link_libraries(mlrisk PUBLIC m)
endif()

# Enable testing
enable_testing()

# Test executable
add_executable(mlrisk_tests
    tests/test_main.c
    tests/test_rolling.c
    tests/test_sizing.c
    tests/test_vol.c
    tests/test_split.c
    tests/test_linreg.c
)

target_link_libraries(mlrisk_tests PRIVATE mlrisk)
target_include_directories(mlrisk_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_test(NAME mlrisk_tests COMMAND $<TARGET_FILE:mlrisk_tests>)

# Example executable
add_executable(vol_target_demo examples/vol_target_demo.c)
target_link_libraries(vol_target_demo PRIVATE mlrisk)
target_include_directories(vol_target_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
