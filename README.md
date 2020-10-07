# MeltPoolDG
## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of this project is to provide solvers for simulating the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are based on continuous and discontinuous finite element methods in an Eulerian setting. For modelling interfaces in the multi-phase flow problem including evaporation, level set methods and phase-field methods will be provided.

This project depends on the following third-party libraries:

- deal.II
- p4est
- Trilinos

![alt text](doc/MeltPoolDG.png?raw=true)

### How to add, build and run a simulation

In the `./include/meltpooldg/simulations` folder you find some example simulations. If you would like to create an additional simulation, e.g. "vortex_bubble", follow the subsequent steps:

```bash
mkdir ./include/meltpooldg/simulations/vortex_bubble
cd include/meltpooldg/simulations/vortex_bubble    
touch vortex_bubble.hpp
touch vortex_bubble.json
```
In the `.hpp` file a child class of the MeltPoolDG::SimulationBase<dim> class must be created. In the `.json`-file the parameters will be specified. Note that the `.json`-file is a command line argument and is only needed at run-time of the simulation. 
The new simulation has to be added to the simulation factory `./include/meltpooldg/simulation_selector.hpp` 
```cpp
else if( simulation_name == "vortex_bubble" )
{
    return std::make_shared<VortexBubble::Simulation<dim>>(parameterfile,
                                                 mpi_communicator);
}
```
You can build an run your simulation (with e.g. 4 processes) using the following commands:
   
```bash  
mkdir build
cd build
cmake -D DEAL_II_DIR=/dealii_build_dir ../.
```
For the debug version call
```bash  
make debug
```
else call
```bash  
make release
```
Then build the code and run the simulation. As an example the simulation of the newly created "vortex_bubble" is demonstrated using 4 threads:
```bash  
make -j 4 
mpirun -np 4 ./run_simulation ../include/meltpooldg/simulations/vortex_bubble.json
```




