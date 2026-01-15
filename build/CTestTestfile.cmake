# CMake generated Testfile for 
# Source directory: C:/MLLibraryRisk
# Build directory: C:/MLLibraryRisk/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(mlrisk_tests "C:/MLLibraryRisk/build/Debug/mlrisk_tests.exe")
  set_tests_properties(mlrisk_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/MLLibraryRisk/CMakeLists.txt;62;add_test;C:/MLLibraryRisk/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(mlrisk_tests "C:/MLLibraryRisk/build/Release/mlrisk_tests.exe")
  set_tests_properties(mlrisk_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/MLLibraryRisk/CMakeLists.txt;62;add_test;C:/MLLibraryRisk/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(mlrisk_tests "C:/MLLibraryRisk/build/MinSizeRel/mlrisk_tests.exe")
  set_tests_properties(mlrisk_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/MLLibraryRisk/CMakeLists.txt;62;add_test;C:/MLLibraryRisk/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(mlrisk_tests "C:/MLLibraryRisk/build/RelWithDebInfo/mlrisk_tests.exe")
  set_tests_properties(mlrisk_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/MLLibraryRisk/CMakeLists.txt;62;add_test;C:/MLLibraryRisk/CMakeLists.txt;0;")
else()
  add_test(mlrisk_tests NOT_AVAILABLE)
endif()
