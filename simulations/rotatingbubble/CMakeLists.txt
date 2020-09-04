set(projectName 
    rotatingbubble
    )
message("--------- building project: ${projectName}")
ADD_EXECUTABLE(${projectName} 
               rotatingbubble.cc
               )
DEAL_II_SETUP_TARGET(${projectName} DEBUG)
TARGET_LINK_LIBRARIES(${projectName} ${CMAKE_PROJECT_NAME})

