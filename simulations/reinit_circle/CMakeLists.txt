set(projectName 
    reinit_circle
    )
message("--------- building project: ${projectName}")
ADD_EXECUTABLE(${projectName} 
               reinit_circle.cpp
               )
DEAL_II_SETUP_TARGET(${projectName})
TARGET_LINK_LIBRARIES(${projectName} ${CMAKE_PROJECT_NAME})
