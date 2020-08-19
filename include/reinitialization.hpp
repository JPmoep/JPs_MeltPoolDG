/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace LevelSetParallel
{
  using namespace dealii; 

  /*
   *    Data for reinitialization of level set equation
   */
  
  typedef enum {olsson2007, undefined} ReinitModelType;
  
  struct ReinitializationData
  {
    ReinitializationData()
        : reinit_model(ReinitModelType::undefined)
        , d_tau(0.01)
        , constant_epsilon(0.0)
    {
    }

    // enum which reinitialization model should be solved
    ReinitModelType reinit_model;
    
    // time step for reinitialization
    double d_tau;
    
    // choose a constant, not cell-size dependent smoothing parameter
    double constant_epsilon;

    // @ add lambda function for calculating epsilon
  }
  
  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the  level set equation
   */
  
  template <int dim>
  class Reinitialization
  {
  private:
    typedef LA::MPI::Vector VectorType;
    typedef LA::MPI::BlockVector BlockVectorType;
    typedef LA::MPI::SparseMatrix SparseMatrixType;
    typedef parallel::distributed::Triangulation<dim> TriangulationType;
    typedef Tensor<1, dim, VectorizedArray<Number>> tensor;
  public:

    /*
     *  Constructor
     */
    Reinitialization();

    void
    initialize( const ReinitializationData &     data_in,
                TriangulationType&               triangulation_in,
                TensorFunction<1, dim>&          advection_field_in,
                const MPI_Comm&                  mpi_commun);

    /*
     *  This function reinitializes the solution of the level set equation for a given solution
     */
    void 
    solve( const VectorType & solution_in,
           VectorType & solution_out );
  private:
    /* Olsson, Kreiss, Zahedi (2007) model 
     *
     * for reinitialization of the level set equation 
     * 
     * @todo: write equation
     */
    void 
    solve_olsson_model( const VectorType & solution_in,
                             VectorType & solution_out );
  };
} // namespace LevelSetParallel
