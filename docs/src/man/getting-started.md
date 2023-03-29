# Getting Started

Suppose the system of nonlinear equations of interest can be described as follows:

```@example getting-started
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
nothing # hide
```

To solve the above equations as a root-finding problem,
we specify the `Hybrid` algorithm by passing either
`Hybrid` or `Hybrid{RootFinding}` as the first argument:

```@example getting-started
# Evaluate Jacobians via finite differencing methods from FiniteDiff.jl
solve(Hybrid, f!, x0)
# Use user-specified Jacobian function and separate out the initialization step
s = init(Hybrid, f!, j!, x0)
solve!(s)
nothing # hide
```

The last line from above calls a non-allocating method `solve!`
that mutates the pre-allocated problem `s` in-place.
On Julia REPL, the essential information is summarized as follows:

```@repl getting-started
s
```

The solution can be retrieved by accessing the corresponding field:

```@repl getting-started
s.x
```

To solve a least-squares problem, specify the algorithm as `Hybrid{LeastSquares}` instead:

```@repl getting-started
s = solve(Hybrid{LeastSquares}, f!, x0)
```

Notice that the gradient norm is `NaN`.
For this specific problem, convergence is attained
before the gradient is ever evaluated.

!!! note

    A root-finding algorithm requires that the number of variables
    matches the number of equations.
    That is, the associated Jacobian matrix must be a square matrix.
    In contrast, a least-squares algorithm does not impose this restriction.

To inspect the solver iteration, summary information can be printed for each evaluation:

```@repl getting-started
s = solve(Hybrid{LeastSquares}, f!, x0, showtrace=1);
```

Notice that all relevant information is collected in the same object:

```@docs
NonlinearSystem
```

Instead of calling `solve` or `solve!`,
which simply iterates `NonlinearSystem` in a loop,
one may manually iterate the solver steps as follows:

```@repl getting-started
s = init(Hybrid{LeastSquares}, f!, x0);
s.solver
iterate(s)
s.solver
```
