##
#  CMake script
##

PROJECT(simulations)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )
ADD_SUBDIRECTORY(error_vortexbubble_parallel)

