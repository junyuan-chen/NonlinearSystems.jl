abstract type ProblemType end

struct RootFinding <: ProblemType end
struct LeastSquares <: ProblemType end

problemtypestr(::Type{RootFinding}) = "Root finding"
problemtypestr(::Type{LeastSquares}) = "Least squares"

abstract type AbstractAlgorithm{T<:ProblemType} end

abstract type AbstractSolver{T<:Number} end

getsolverstate(s::AbstractSolver) = getfield(s, :state)[]
getiter(s::AbstractSolver) = getfield(getsolverstate(s), :iter)
getfnorm(s::AbstractSolver) = getfield(getsolverstate(s), :fnorm)
getpnorm(s::AbstractSolver) = getfield(getsolverstate(s), :pnorm)
getgrad(s::AbstractSolver) = getfield(s, :grad)
getlinsolver(s::AbstractSolver) = getfield(s, :linsolver)

@enum SolverIterationState::Int8 begin
    normal
    eval_noprogress
    jac_noprogress
    maxntrial_reached
end

@enum SolverExitState::Int8 begin
    ftol_reached
    gtol_reached
    xtol_reached
    xtolr_reached
    maxiter_reached
    inprogress
    failed
end

struct NonlinearSystem{P<:ProblemType, V, M, S<:AbstractSolver,
        LB<:Union{AbstractVector, Nothing}, UB<:Union{AbstractVector, Nothing}}
    fdf::OnceDifferentiable{V, M, V}
    x::V
    fx::V
    dx::V
    lb::LB
    ub::UB
    solver::S
    maxiter::Int
    ftol::Float64
    gtol::Float64
    xtol::Float64
    xtolr::Float64
    state::RefValue{Tuple{SolverIterationState, SolverExitState}}
    showtrace::Int
end

"""
    NonlinearSystem(::Type{P}, fdf::OnceDifferentiable, x, fx, dx, solver; kwargs...)

Construct a `NonlinearSystem` for holding all information
used for solving a nonlinear system of equations with problem type `P`.
Users are not expected to use this method directly
but should instead call `init` or `solve` to generate the problem.
Any keyword argument passed to `init` or `solve` that is not accepted by
a specific solution algorithm is passed to the constructor of `NonlinearSystem`.
For the relevant solution algorithms, see [`Hybrid`](@ref).

# Keywords
- `lower::Union{AbstractVector, Nothing}=nothing`: element-wise lower bounds
  for solution candidates.
- `upper::Union{AbstractVector, Nothing}=nothing`: element-wise upper bounds
  for solution candidates.
- `maxiter::Integer=1000`: maximum number of iteration allowed before terminating.
- `ftol::Real=1e-8`: absolute tolerance for the infinity norm of residuals `fx`.
- `gtol::Real=1e-10`: absolute tolerance for the infinity norm of gradient vector;
  only relevant for solving least squares.
- `xtol::Real=0.0`: absolute tolerance for the infinity norm of a step `dx`.
- `xtolr::Real=0.0`: relative tolerance for the infinity norm of a step `dx`
  as a proportion of `x`.
- `showtrace::Union{Bool,Integer}=false`: print summary information for each trial
  made by the solver; with `showtrace=true`, information is printed
  once every 20 iterations; an interger value specifies the gap for printing.
"""
function NonlinearSystem(::Type{P}, fdf::OnceDifferentiable{V,M,V}, x::AbstractVector,
        fx::AbstractVector, dx::AbstractVector, solver::AbstractSolver;
        lower::Union{AbstractVector, Nothing}=nothing,
        upper::Union{AbstractVector, Nothing}=nothing,
        maxiter::Integer=1000, ftol::Real=1e-8, gtol::Real=1e-10,
        xtol::Real=0.0, xtolr::Real=0.0, showtrace::Union{Bool,Integer}=false) where {P,V,M}
    if lower !== nothing
        length(lower) == length(dx) || throw(DimensionMismatch(
            "length of lower is expected to be $(length(dx))"))
        all(i->x[i]>=lower[i], eachindex(lower)) || throw(ArgumentError(
            "initial values cannot be smaller than lower bounds"))
    end
    if upper !== nothing
        length(upper) == length(dx) || throw(DimensionMismatch(
            "length of upper is expected to be $(length(dx))"))
        all(i->x[i]<=upper[i], eachindex(upper)) || throw(ArgumentError(
            "initial values cannot be greater than upper bounds"))
    end
    showtrace === true && (showtrace = 20)
    showtrace === false && (showtrace = 0)
    return NonlinearSystem{P, V, M, typeof(solver), typeof(lower), typeof(upper)}(
        fdf, x, fx, dx, lower, upper, solver,
        convert(Int, maxiter), convert(Float64, ftol), convert(Float64, gtol),
        convert(Float64, xtol), convert(Float64, xtolr),
        Ref((normal, inprogress)), showtrace)
end

nvar(s::NonlinearSystem) = length(s.fdf.x_f)
nequ(s::NonlinearSystem) = length(s.fdf.F)
size(s::NonlinearSystem) = (nvar(s), nequ(s))
size(s::NonlinearSystem, dim::Integer) =
    dim == 1 ? nequ(s) : dim == 2 ? nvar(s) : throw(ArgumentError("dim can only be 1 or 2"))

getiter(s::NonlinearSystem) = getiter(s.solver)
getlinsolver(s::NonlinearSystem) = getlinsolver(s.solver)
getiterstate(s::NonlinearSystem) = s.state[][1]
getexitstate(s::NonlinearSystem) = s.state[][2]

@inline _test_ftol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(s.fx[i]) < s.ftol)

@inline _test_gtol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(getgrad(s.solver)[i] * max(s.x[i], 1)) < s.gtol)

@inline _test_xtol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(s.dx[i]) < s.xtol + s.xtolr * abs(s.x[i]))

function assess_state(s::NonlinearSystem{P}, iterstate::SolverIterationState) where P
    if iterstate !== normal
        return failed
    elseif getiter(s) >= s.maxiter
        return maxiter_reached
    elseif s.ftol > 0 && all(Fix1(_test_ftol_i, s), eachindex(s.fx))
        return ftol_reached
    elseif P === LeastSquares && s.gtol > 0 && all(Fix1(_test_gtol_i, s), eachindex(s.x))
        return gtol_reached
    elseif s.xtol > 0 && all(Fix1(_test_xtol_i, s), eachindex(s.dx))
        return xtol_reached
    else
        return inprogress
    end
end

@inline function iterate(s::NonlinearSystem, state=(1, normal, inprogress))
    # How iter changes depends on the specific algorithm
    iter, iterstate = s.solver(s.fdf, s.x, s.fx, s.dx, s.lb, s.ub)
    exitstate = assess_state(s, iterstate)
    s.state[] = (iterstate, exitstate)
    return s, (iter, iterstate, exitstate) # Termination is never enforced here
end

function solve!(s::NonlinearSystem)
    iter = 1
    iterst, exitst = s.state[]
    while exitst === inprogress
        s.showtrace > 0 && iszero((iter-1) % s.showtrace) &&
            _show_trace(stdout, s.solver, true)
        s, (iter, iterst, exitst) = iterate(s, (iter, iterst, exitst))
    end
    s.showtrace > 0 && !(iszero((iter-1) % s.showtrace)) &&
        _show_trace(stdout, s.solver, true)
    return s
end

solve!(s::NonlinearSystem, x0; kwargs...) = solve!(init(s, x0; kwargs...))

init(algo::AbstractAlgorithm, args...; kwargs...) =
    init(typeof(algo), args...; kwargs...)

init(Algo::Type{<:AbstractAlgorithm}, f, x0::AbstractVector; kwargs...) =
    init(Algo, OnceDifferentiable(f, similar(x0), similar(x0)), x0; kwargs...)

init(Algo::Type{<:AbstractAlgorithm}, f, j, x0::AbstractVector; kwargs...) =
    init(Algo, OnceDifferentiable(f, j, similar(x0), similar(x0)), x0; kwargs...)

function show(io::IO, s::NonlinearSystem{P}) where P
    print(io, nequ(s), '×', nvar(s), ' ', typeof(s).name.name, '{', P, "}(")
    print(io, algorithmtype(s.solver), ", ", getfnorm(s.solver), ", ", getexitstate(s), ')')
end

function show(io::IO, ::MIME"text/plain", s::NonlinearSystem{P}) where P
    println(io, nequ(s), '×', nvar(s), ' ', typeof(s), ':')
    w = 32
    println(io, rpad("  Problem type: ", w), problemtypestr(P))
    println(io, rpad("  Algorithm: ", w), algorithmtype(s.solver))
    println(IOContext(io, :compact=>true), rpad("  Candidate (x): ", w), s.x)
    println(io, rpad("  ‖f(x)‖₂: ", w), getfnorm(s.solver))
    P === LeastSquares && println(io, rpad("  ‖∇f(x)‖₂: ", w), enorm(getgrad(s.solver)))
    println(io, rpad("  ‖Ddx‖₂: ", w), getpnorm(s.solver))
    println(io, rpad("  Solver exit state: ", w), getexitstate(s))
    println(io, rpad("  Iterations: ", w), getiter(s.solver))
    println(io, rpad("  Residual calls (f): ", w), s.fdf.f_calls[1])
    print(io, rpad("  Jacobian calls (df/dx): ", w), s.fdf.df_calls[1])
end
