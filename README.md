# MeltPoolDG
## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of this project is to provide solvers for simulating the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are based on continuous and discontinuous finite element methods in an Eulerian setting. For modelling interfaces in the multi-phase flow problem including evaporation, level set methods and phase-field methods will be provided.

This project depends on the following third-party libraries:

- deal.II
- p4est
- Trilinos

![alt text](doc/MeltPoolDG.png?raw=true)

### How to add and run a new simulation

In the ./simulations folder you find some example simulations. If you would like to create a simulation "vortex_bubble", follow the subsequent steps:

```bash
cd simulations
echo "ADD_SUBDIRECTORY(vortex_bubble)" >> CMakeLists.txt
mkdir vortex_bubble
cd vortex_bubble    
touch vortex_bubble.cpp
touch vortex_bubble.json
cp ../rotating_bubble/CMakeLists .
```
   
In the CMakeLists file change the project name and the *.cpp-file name containing the main function. You can build an run your simulation (with e.g. 4 processes) using the following commands:
   
```bash  
mkdir build
cd build
cmake -D DEAL_II_DIR=/myDealIIBuildDir ../.
```
Change to the simulations director
```bash  
cd simulations
```
For the debug version call
```bash  
make debug
```
else call
```bash  
make release
```
Then build the code and run the simulation
```bash  
make -j 4 
mpirun -np 4 ./run_simulation folder_of_your_json_file/your_input_file.json
```




