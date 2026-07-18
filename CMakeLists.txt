cmake_minimum_required(VERSION 3.22)

set(Project_name Ungulate)
set(Elf_name ungulate)
project(${Project_name})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

if(CMAKE_BUILD_TYPE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
else()
  set(BUILD_TYPE "debug")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

if(BUILD_TYPE STREQUAL "debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif(BUILD_TYPE STREQUAL "debug")

if(BUILD_TYPE STREQUAL "release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(BUILD_TYPE STREQUAL "release")

find_package(nlohmann_json REQUIRED)
if(nlohmann_json_FOUND)
  message("Found nlohmann::json")
endif()

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message("Found OpenCV")
endif()

find_package(OpenSSL REQUIRED)
if(OpenSSL_FOUND)
  message("Found OpenSSL")
endif()

find_package(SQLite3 REQUIRED)
if(SQLite3_FOUND)
  message("Found SQLite3")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

# Core srcs
file(GLOB Srcs_top
  *.cpp
  database/*.cpp
  llm/*.cpp
  log/*.cpp
  file/*.cpp
  query/*.cpp
  server/*.cpp)

# Build ELF executable
set(Srcs ${Srcs_top})
add_executable(${Elf_name} ${Srcs})

target_link_libraries(${Elf_name} LINK_PUBLIC crypto)
target_link_libraries(${Elf_name} LINK_PUBLIC ggml)
target_link_libraries(${Elf_name} LINK_PUBLIC ggml-base)
target_link_libraries(${Elf_name} LINK_PUBLIC ggml-cuda)
target_link_libraries(${Elf_name} LINK_PUBLIC gs)
target_link_libraries(${Elf_name} LINK_PUBLIC llama)
target_link_libraries(${Elf_name} LINK_PUBLIC mtmd)
target_link_libraries(${Elf_name} LINK_PUBLIC ${OpenCV_LIBS})
target_link_libraries(${Elf_name} LINK_PUBLIC poppler-cpp)
target_link_libraries(${Elf_name} LINK_PUBLIC sqlite3)
target_link_libraries(${Elf_name} LINK_PUBLIC ssl)

install(FILES
  ${CMAKE_CURRENT_LIST_DIR}/plugins/vec0.so
  DESTINATION /usr/local/share/ungulate/plugins)
install(DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR}/client
  DESTINATION /usr/local/share/ungulate)
install(TARGETS ${Elf_name} DESTINATION bin)
