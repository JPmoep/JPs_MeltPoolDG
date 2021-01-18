##
#  CMake script
##

MESSAGE("_________________________________________________________________________________")
MESSAGE("                                                                                 ")
MESSAGE("                              MeltPoolDG                                         ")
MESSAGE("                                                                                 ")
MESSAGE("_________________________________________________________________________________")


# this is the standard deal.II search mechanism, including check for Trilinos and p4est
cmake_minimum_required(VERSION 3.3)  
set(CMAKE_CXX_STANDARD 17)

## avoid in-source compilation
set(DEFAULT_OUT_OF_SOURCE_FOLDER "build")
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Change to ${DEFAULT_OUT_OF_SOURCE_FOLDER} folder and run then cmake
     ")
endif ()

FIND_PACKAGE(deal.II 9.2.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

MESSAGE("-- deal.II found: ${DEAL_II_DIR}")

IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "\nThis library requires a deal.II installation built with support for p4est but it appears to be missing!\n"
    )
ENDIF()

IF(NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(FATAL_ERROR
    "\nThis library requires a deal.II installation built with support for Trilinos but it appears to be missing!\n"
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set the source files to be compiled
# temporarilly disabled as long as no source files in meltpooldg exists
SET(TARGET_SRC meltpooldg.cpp)

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include ${CMAKE_CURRENT_BINARY_DIR}/include)

PROJECT(MeltPoolDG)

#ADD_LIBRARY(MeltPoolDG ${TARGET_SRC} .)

FIND_LIBRARY(ADAFLO adaflo HINTS ${ADAFLO_LIB})
IF(ADAFLO)
  MESSAGE("-- Adaflo found: ${ADAFLO}") 
  TARGET_COMPILE_DEFINITIONS(${projectName} PUBLIC MELT_POOL_DG_WITH_ADAFLO)
  TARGET_INCLUDE_DIRECTORIES(${projectName} PUBLIC ${ADAFLO_INCLUDE})
  TARGET_LINK_LIBRARIES(${projectName} ${ADAFLO})
  TARGET_LINK_LIBRARIES(${projectName} ${ADAFLO})
  SET(DEAL_II_WITH_ADAFLO_SUPPORT ON )
ENDIF()

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  MESSAGE("-- tests found")
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

SET(projectName "my")

message("-- building project: ${CMAKE_PROJECT_NAME}")
ADD_EXECUTABLE(${projectName} 
                meltpooldg.cpp)

DEAL_II_SETUP_TARGET(${projectName})

TARGET_LINK_LIBRARIES(${projectName})

#tests
SET(_n_tests "0")
enable_testing()
ADD_SUBDIRECTORY(tests)


