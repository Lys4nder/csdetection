cmake_minimum_required(VERSION 3.8)
project(Image LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify paths for dependencies
list(APPEND CMAKE_PREFIX_PATH "/root/.local")

# Sets to autogenerate the MOC files
set(CMAKE_AUTOMOC ON)

find_package(Threads REQUIRED)

# Find Qt6 components
find_package(Qt6 COMPONENTS Widgets Core Gui REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)


# Check if Qt6 was found
if (Qt6_FOUND)
  message(STATUS "Qt6 was found")
else()
  message(FATAL_ERROR "Qt6 was not found")
endif()


# Check if OpenCV was found
if (OpenCV_FOUND)
  message(STATUS "OpenCV ${OpenCV_VERSION} was found")
else()
  message(FATAL_ERROR "OpenCV was not found")
endif()

include_directories(${OpenCV_INCLUDE_DIRS}
)


# Sets the project sources file
set(PROJECT_SOURCES
  main.cc
)

# Sets the project headers file
set(PROJECT_HEADERS
)

# Sets the executable
add_executable(${CMAKE_PROJECT_NAME}
  ${PROJECT_SOURCES}
  ${PROJECT_HEADERS}
)

# Links the executable against the necessary dependencies
target_link_libraries(${CMAKE_PROJECT_NAME}
  Qt6::Widgets
  Qt6::Core
  ${OpenCV_LIBS}
)

# Sets a library with project file for testing
add_library(${CMAKE_PROJECT_NAME}_lib
  ${PROJECT_SOURCES}
  ${PROJECT_HEADERS}
)

# Links the library against the necessary dependencies
target_link_libraries(${CMAKE_PROJECT_NAME}_lib
  Qt6::Widgets
  Qt6::Core
  ${OpenCV_LIBS}
)

# Adds the test directory

