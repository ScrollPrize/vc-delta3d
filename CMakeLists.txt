cmake_minimum_required(VERSION 3.22)

project(vc_delta3d VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(VC_DELTA3D_BUILD_PYTHON "Build the Python extension" OFF)
option(VC_DELTA3D_BUILD_TESTS "Build the C++ codec tests" ${PROJECT_IS_TOP_LEVEL})
option(VC_DELTA3D_INSTALL_CPP "Install the C++ library and CMake package" ON)

if(NOT TARGET zstd::libzstd)
    find_package(zstd CONFIG QUIET)
    if(TARGET zstd::libzstd_shared)
        add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
    elseif(TARGET zstd::libzstd_static)
        add_library(zstd::libzstd ALIAS zstd::libzstd_static)
    endif()
endif()

if(NOT TARGET zstd::libzstd)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(VC_DELTA3D_ZSTD REQUIRED IMPORTED_TARGET libzstd)
    add_library(zstd::libzstd ALIAS PkgConfig::VC_DELTA3D_ZSTD)
endif()

add_library(vc_delta3d STATIC src/codec.cpp)
add_library(vc_delta3d::vc_delta3d ALIAS vc_delta3d)
set_target_properties(vc_delta3d PROPERTIES
    EXPORT_NAME vc_delta3d
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(vc_delta3d PUBLIC cxx_std_20)
target_include_directories(vc_delta3d
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(vc_delta3d PUBLIC
    $<BUILD_INTERFACE:zstd::libzstd>
    $<INSTALL_INTERFACE:zstd::libzstd>
)

if(VC_DELTA3D_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE VC_DELTA3D_NANOBIND_CMAKE_DIR
        RESULT_VARIABLE VC_DELTA3D_NANOBIND_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(VC_DELTA3D_NANOBIND_RESULT EQUAL 0 AND VC_DELTA3D_NANOBIND_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${VC_DELTA3D_NANOBIND_CMAKE_DIR}")
    endif()
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(vc_delta3d_python NOMINSIZE python/bindings.cpp)
    set_target_properties(vc_delta3d_python PROPERTIES OUTPUT_NAME _codec)
    target_link_libraries(vc_delta3d_python PRIVATE vc_delta3d::vc_delta3d)
    target_compile_features(vc_delta3d_python PRIVATE cxx_std_20)
    install(TARGETS vc_delta3d_python
        LIBRARY DESTINATION vc_delta3d
        RUNTIME DESTINATION vc_delta3d
    )
endif()

if(VC_DELTA3D_BUILD_TESTS)
    enable_testing()
    add_executable(vc_delta3d_tests tests/test_codec.cpp)
    target_include_directories(vc_delta3d_tests PRIVATE tests)
    target_link_libraries(vc_delta3d_tests PRIVATE vc_delta3d::vc_delta3d)
    add_test(NAME vc_delta3d_tests COMMAND vc_delta3d_tests)
endif()

if(VC_DELTA3D_INSTALL_CPP)
    install(TARGETS vc_delta3d
        EXPORT vc_delta3dTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT vc_delta3dTargets
        FILE vc_delta3dTargets.cmake
        NAMESPACE vc_delta3d::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_delta3d
    )

    configure_package_config_file(
        cmake/vc_delta3dConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/vc_delta3dConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_delta3d
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/vc_delta3dConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/vc_delta3dConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/vc_delta3dConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_delta3d
    )
endif()
