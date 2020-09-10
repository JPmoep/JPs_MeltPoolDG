set(projectName 
    advection_diffusion
    )
message("--------- building project: ${projectName}")
ADD_EXECUTABLE(${projectName} 
               advection_diffusion.cpp
               )
DEAL_II_SETUP_TARGET(${projectName})
TARGET_LINK_LIBRARIES(${projectName} ${CMAKE_PROJECT_NAME})
