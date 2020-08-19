set(projectName 
    error_vortexbubble_parallel
    )
message("--------- building project: ${projectName}")
ADD_EXECUTABLE(${projectName} 
               vortexbubble_parallel.cpp
               )
DEAL_II_SETUP_TARGET(${projectName} DEBUG)
TARGET_LINK_LIBRARIES(${projectName} ${CMAKE_PROJECT_NAME})

