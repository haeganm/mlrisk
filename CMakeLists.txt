cmake_minimum_required(VERSION 3.15)
project(mlrisk VERSION 2.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /WX")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Werror")
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

# Install / export so find_package(mlrisk) works
add_library(mlrisk::mlrisk ALIAS mlrisk)
include(GNUInstallDirs)
install(TARGETS mlrisk EXPORT mlriskTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mlriskTargets NAMESPACE mlrisk::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlrisk)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfig.cmake
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/mlriskTargets.cmake\")\n")
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mlriskConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlrisk)
