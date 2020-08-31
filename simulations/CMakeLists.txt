##
#  CMake script
##

PROJECT(simulations)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )
#ADD_SUBDIRECTORY(error_vortexbubble_parallel)
ADD_SUBDIRECTORY(test_rotatingbubble)
#ADD_SUBDIRECTORY(helloworld)
#ADD_SUBDIRECTORY(newFrameWorkTest)

