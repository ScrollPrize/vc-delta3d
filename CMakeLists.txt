cmake_minimum_required(VERSION 3.22)

project(vc_delta3d VERSION 0.4.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(VC_DELTA3D_BUILD_PYTHON "Build the Python extension" OFF)
option(VC_DELTA3D_BUILD_TESTS "Build the C++ codec tests" ${PROJECT_IS_TOP_LEVEL})
option(VC_DELTA3D_INSTALL_CPP "Install the C++ library and CMake package" ON)
option(VC_DELTA3D_MODERN_SMOKE "Build no-dependency integration smoke test" OFF)
if(VC_DELTA3D_BUILD_PYTHON OR VC_DELTA3D_BUILD_TESTS OR VC_DELTA3D_MODERN_SMOKE)
    enable_language(CXX)
endif()

add_library(vc_delta3d STATIC)
option(VC_DELTA3D_BUILD_NG "Build modern NG/Delta presets" ON)
if(VC_DELTA3D_BUILD_NG)
    include(cmake/modern.cmake)
endif()
target_sources(vc_delta3d PRIVATE src/delta_filter.c src/rans.c src/legacy.c src/api.c)
if(VC_DELTA3D_BUILD_TESTS OR VC_DELTA3D_MODERN_SMOKE)
    enable_testing()
    add_executable(vc_delta3d_filter_tests tests/test_delta_filter.c src/delta_filter.c)
    if(NOT MSVC)
        target_link_libraries(vc_delta3d_filter_tests PRIVATE m)
        target_compile_options(vc_delta3d_filter_tests PRIVATE -UNDEBUG -ffp-contract=off)
    else()
        target_compile_options(vc_delta3d_filter_tests PRIVATE /UNDEBUG)
    endif()
    add_test(NAME vc_delta3d_filter_tests COMMAND vc_delta3d_filter_tests)
    add_executable(vc_delta3d_rans_tests tests/test_rans.c src/rans.c)
    target_include_directories(vc_delta3d_rans_tests PRIVATE src)
    if(NOT MSVC)
        target_compile_options(vc_delta3d_rans_tests PRIVATE -UNDEBUG)
    else()
        target_compile_options(vc_delta3d_rans_tests PRIVATE /UNDEBUG)
    endif()
    add_test(NAME vc_delta3d_rans_tests COMMAND vc_delta3d_rans_tests)
    add_executable(vc_delta3d_legacy_tests tests/test_legacy.c src/legacy.c src/rans.c src/delta_filter.c)
    target_include_directories(vc_delta3d_legacy_tests PRIVATE src)
    if(NOT MSVC)
        target_compile_options(vc_delta3d_legacy_tests PRIVATE -UNDEBUG -ffp-contract=off)
        target_link_libraries(vc_delta3d_legacy_tests PRIVATE m)
    else()
        target_compile_options(vc_delta3d_legacy_tests PRIVATE /UNDEBUG)
    endif()
    add_test(NAME vc_delta3d_legacy_tests COMMAND vc_delta3d_legacy_tests)
endif()
add_library(vc_delta3d::vc_delta3d ALIAS vc_delta3d)
set_target_properties(vc_delta3d PROPERTIES
    EXPORT_NAME vc_delta3d
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(vc_delta3d PUBLIC c_std_11)
if(CMAKE_CXX_COMPILER_LOADED)
    target_compile_features(vc_delta3d INTERFACE cxx_std_20)
endif()
# The NG fit solves its normal equations in double precision; no fused
# multiply-adds keeps the fitted weights, and so the payload bytes, identical
# across platforms (GCC contracts by default on aarch64).
if(MSVC AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vc_delta3d PRIVATE /clang:-ffp-contract=off)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vc_delta3d PRIVATE -ffp-contract=off)
endif()
target_include_directories(vc_delta3d
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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

    nanobind_add_module(vc_delta3d_python NOMINSIZE FREE_THREADED python/bindings.cpp)
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
    # doctest is a single header; SOURCE_SUBDIR skips its own CMake project,
    # whose old cmake_minimum_required newer CMake versions reject.
    include(FetchContent)
    FetchContent_Declare(doctest
        URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
        URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SOURCE_SUBDIR header-only
    )
    FetchContent_MakeAvailable(doctest)
    add_library(vc_delta3d_doctest INTERFACE)
    target_include_directories(vc_delta3d_doctest SYSTEM INTERFACE ${doctest_SOURCE_DIR})

    add_executable(vc_delta3d_tests tests/test_codec.cpp)
    target_link_libraries(vc_delta3d_tests PRIVATE vc_delta3d::vc_delta3d vc_delta3d_doctest)
    add_test(NAME vc_delta3d_tests COMMAND vc_delta3d_tests)

endif()

if(VC_DELTA3D_INSTALL_CPP)
    install(FILES LICENSE AUTHORS.md DESTINATION ${CMAKE_INSTALL_DATADIR}/vc-delta3d)
    install(FILES src/ng/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/vc-delta3d RENAME LICENSE-NG)
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
