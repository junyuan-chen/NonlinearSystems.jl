# NonlinearSystems.jl

*Solve nonlinear systems of equations and nonlinear least squares in Julia*

[![CI-stable][CI-stable-img]][CI-stable-url]
[![codecov][codecov-img]][codecov-url]
[![PkgEval][pkgeval-img]][pkgeval-url]
[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]

[CI-stable-img]: https://github.com/junyuan-chen/NonlinearSystems.jl/workflows/CI-stable/badge.svg
[CI-stable-url]: https://github.com/junyuan-chen/NonlinearSystems.jl/actions?query=workflow%3ACI-stable

[codecov-img]: https://codecov.io/gh/junyuan-chen/NonlinearSystems.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/junyuan-chen/NonlinearSystems.jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/N/NonlinearSystems.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/N/NonlinearSystems.html

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://junyuan-chen.github.io/NonlinearSystems.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://junyuan-chen.github.io/NonlinearSystems.jl/dev/

[NonlinearSystems.jl](https://github.com/junyuan-chen/NonlinearSystems.jl)
is a Julia package for solving nonlinear systems of equations and nonlinear least squares.
It renovates well-trusted solution algorithms with
highly performant and extensible implementation in native Julia language.

NonlinearSystems.jl places special emphasis on

- Low number of evaluations needed for updating the Jacobian matrix
- Flexibility of swapping the underlying linear solver based on array type and hardware
- Zero memory allocation incurred in iteration steps

At this moment, the only solution algorithm implemented
is a modified version of Powell's hybrid method
(a trust region method with dogleg).
Relations to existing packages are further discussed towards the end.

## Quick Start

NonlinearSystems.jl adopts the
[CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) interface
and wraps a residual function as `OnceDifferentiable` defined in
[NLSolversBase.jl](https://github.com/JuliaNLSolvers/NLSolversBase.jl)
with an optionally user-provided Jacobian function.
The same interface is shared for
solving a root-finding problems and solving a least-squares problems.

```julia
using NonlinearSystems

# Residual function
function f!(F, x)
    F[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    F[2] = sin(x[2] * exp(x[1]) - 1)
    return F
end

# Jacobian function (optional)
function j!(J, x)
    J[1,1] = x[2]^3 - 7
    J[1,2] = 3 * x[2]^2 * (x[1] + 3)
    u = exp(x[1]) * cos(x[2] * exp(x[1]) - 1)
    J[2,1] = x[2] * u
    J[2,2] = u
    return J
end

# Initial value
x0 = [0.1, 1.2]

# Evaluate Jacobians via finite differencing methods from FiniteDiff.jl
solve(Hybrid{RootFinding}, f!, x0)

# Use user-specified Jacobian function and separate out the initialization step
s = init(Hybrid{LeastSquares}, f!, j!, x0)
solve!(s)
```

For more details, please see the [documentation][docs-stable-url].

## Related Packages

NonlinearSystems.jl addresses the following limitations that the related packages do not:

- [MINPACK.jl](https://github.com/sglyon/MINPACK.jl) and [GSL.jl](https://github.com/JuliaMath/GSL.jl)
  - No option for swapping the linear solver
  - Use of rank-1 update of the Jacobian matrix and factorization cannot be adjusted
  - Wrappers of C interface; no native Julia implementation
  - MINPACK.jl does not work on Apple Silicon
- [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)
  - No option for swapping the linear solver
  - No rank-1 update of the Jacobian matrix and factorization
  - Iteration steps are not non-allocating
  - No support for nonlinear least squares
- [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl)
  - Trust region solver (`TrustRegion`) does not provide correct answers on test problems (as of version 1.6)
  - No rank-1 update of the Jacobian matrix and factorization
  - Iteration steps are not non-allocating
  - No support for nonlinear least squares
- [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl)
  - Only solves nonlinear least squares
  - No rank-1 update of the Jacobian matrix and factorization
- [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl)
  - Only solves nonlinear least squares
  - Performance seems to be dominated by LeastSquaresOptim.jl

## Roadmap

The development of NonlinearSystems.jl is still in an early stage.
At this moment, only trust-region methods are considered and
the linear problem involved in each iteration is only solved by dense matrix factorization.

The following features will be added in future:

- Support for sparse Jacobian matrices
- Support for conducting linear algebra on GPUs

## References

**Moré, Jorge J., Danny C. Sorenson, Burton S. Garbow, and Kenneth E. Hillstrom.** 1984.
"The MINPACK Project."
In *Sources and Development of Mathematical Software*,
ed. Wayne R. Cowell, 88-111. New Jersey: Prentice-Hall.

**Nocedal, Jorge, and Stephen J. Wright.** 2006.
*Numerical Optimization.* 2nd ed. New York: Springer.

**Powell, Michael J. D.** 1970.
"A Hybrid Method for Nonlinear Equations."
In *Numerical Methods for Nonlinear Algebraic Equations*,
ed. Philip Rabinowitz, 87-114. London: Gordon and Breach.
