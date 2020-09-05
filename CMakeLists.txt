##
#  CMake script
##

MESSAGE("_________________________________________________________________________________")
MESSAGE("                                                                                 ")
MESSAGE("                              MeltPoolDG                                         ")
MESSAGE("                                                                                 ")
MESSAGE("_________________________________________________________________________________")

# this is the standard deal.II search mechanism, including check for Trilinos and p4est

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

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
FILE(GLOB_RECURSE TARGET_SRC  "source/*.cpp")
SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)

PROJECT(MeltPoolDG)

ADD_LIBRARY(${CMAKE_PROJECT_NAME} ${TARGET_SRC})

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

DEAL_II_SETUP_TARGET(MeltPoolDG)

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

# applications
ADD_SUBDIRECTORY(simulations)

# tests --> will be added asap
#enable_testing()
#ADD_SUBDIRECTORY(tests)


