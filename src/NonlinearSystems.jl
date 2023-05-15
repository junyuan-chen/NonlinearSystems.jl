module NonlinearSystems

using Base: RefValue, Fix1
using CommonSolve: solve
using FastLapackInterface: LUWs
using LinearAlgebra: BLAS, LAPACK, LU, Cholesky, cholesky!, Hermitian, ldiv!, mul!,
    lowrankupdate!
using NLSolversBase: OnceDifferentiable, value_jacobian!!, jacobian!!
using PositiveFactorizations
using Printf

import Base: iterate, size, show
import CommonSolve: init, solve!

# Reexport
export init, solve!, solve
export OnceDifferentiable

export ProblemType,
       RootFinding,
       LeastSquares,
       AbstractAlgorithm,
       AbstractSolver,
       getsolverstate,
       getiter,
       NonlinearSystem,
       getiterstate,
       getexitstate,

       DenseLUSolver,
       DenseCholeskySolver,

       Hybrid,
       HybridSolver

include("utils.jl")
include("interface.jl")
include("linsolve.jl")
include("hybrid.jl")
include("precompile.jl")

end # module NonlinearSystems
