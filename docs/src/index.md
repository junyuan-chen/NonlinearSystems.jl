# NonlinearSystems.jl

Welcome to the documentation site for NonlinearSystems.jl!

[NonlinearSystems.jl](https://github.com/junyuan-chen/NonlinearSystems.jl)
is a Julia package for solving nonlinear systems of equations and nonlinear least squares.
It renovates well-trusted solution algorithms with
highly performant and extensible implementation in native Julia language.

NonlinearSystems.jl places special emphasis on

- Low number of evaluations needed for updating the Jacobian matrix
- Flexibility of swapping the underlying linear solvers based on array type and hardware
- Zero memory allocation incurred in iteration steps

At this moment, the only solution algorithm implemented
is a modified version of Powell's hybrid method
(a trust region method with dogleg).

## Installation

NonlinearSystems.jl can be installed with the Julia package manager
[Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/).
From the Julia REPL, type `]` to enter the Pkg REPL and run:

```
pkg> add NonlinearSystems
```
