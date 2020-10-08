# MeltPoolDG
## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of this project is to provide solvers for simulating the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are based on continuous and discontinuous finite element methods in an Eulerian setting. For modelling interfaces in the multi-phase flow problem including evaporation, level set methods and phase-field methods will be provided.

This project depends on the following third-party libraries:

- deal.II
- p4est
- Trilinos

Before installing deal.II, you have to install the two libraries p4est and Trilinos. More information for that you can find here for p4est (https://www.dealii.org/9.2.0/external-libs/p4est.html) and here for Trilinos (https://www.dealii.org/9.2.0/external-libs/trilinos.html).  
To configure the deal.II library, you have to use now these arguments for cmake:

 ```bash
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir 
-DP4EST_DIR=/path/to/installation -DDEAL_II_WITH_P4EST=ON -DEAL_II_WITH_MPI=ON -DTRILINOS_DIR=/path/to/trilinos -DDEAL_II_WITH_TRILINOS=ON ../deal.II
```
 
then compiling as usual:
 
```bash
make --jobs=4 install
make test
```

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




