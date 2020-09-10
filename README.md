# MeltPoolDG
## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of this project is to provide solvers for simulating the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are based on continuous and discontinuous finite element methods in an Eulerian setting. For modelling interfaces in the multi-phase flow problem including evaporation, level set methods and phase-field methods will be provided.

This project depends on the following third-party libraries:

    * dealii
    * p4est
    * Trilinos

![alt text](doc/MeltPoolDG.png?raw=true)

### How to add and run a new simulation

In the ./simulations folder you find some example simulations. If you would like to create a simulation "vortex_bubble", follow the subsequent steps:

    * cd simulations
    * echo "ADD_SUBDIRECTORY(vortex_bubble)" >> CMakeLists.txt
    * mkdir vortex_bubble
    * cd vortex_bubble    
    * touch vortex_bubble.cpp
    * touch vortex_bubble.json
    * cd ../rotating_bubble/CMakeLists .
   
In the CMakeLists file change the project name and the *.cpp-file name containing the main function. You can build an run your simulation (with e.g. 4 cores) using the folllowing commands:
     
     * cmake -D DEAL_II_DIR=/myDealIIBuildDir ../../.
     * make -j 6 
     * mpirun -np 4 ./vortexbubble





