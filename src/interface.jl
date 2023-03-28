abstract type ProblemType end
struct RootFinding <: ProblemType end
struct LeastSquares <: ProblemType end

problemtypestr(::Type{RootFinding}) = "Root finding"
problemtypestr(::Type{LeastSquares}) = "Least squares"

abstract type AbstractAlgorithm{T<:ProblemType} end

abstract type AbstractSolver{T<:Number} end

getiter(s::AbstractSolver) = getfield(s, :iter)
getgrad(s::AbstractSolver) = getfield(s, :grad)
getfnorm(s::AbstractSolver) = getfield(s, :fnorm)
getpnorm(s::AbstractSolver) = getfield(s, :pnorm)
getlinsolver(s::AbstractSolver) = getfield(s, :linsolver)

@enum SolverExitState::Int8 begin
    ftol_reached
    gtol_reached
    xtol_reached
    xtolr_reached
    maxiter_reached
    inprogress
    failed
end

@enum SolverIterationState::Int8 begin
    normal
    eval_noprogress
    jac_noprogress
    maxntrial_reached
end

mutable struct NonlinearSystem{P<:ProblemType, V, M, S<:AbstractSolver}
    fdf::OnceDifferentiable{V, M, V}
    x::V
    fx::V
    dx::V
    solver::S
    maxiter::Int
    ftol::Float64
    gtol::Float64
    xtol::Float64
    xtolr::Float64
    iterstate::SolverIterationState
    exitstate::SolverExitState
    showtrace::Int
end

function NonlinearSystem(::Type{P}, fdf::OnceDifferentiable{V,M,V}, x::AbstractVector,
        fx::AbstractVector, dx::AbstractVector, solver::AbstractSolver;
        maxiter::Integer=1000, ftol::Real=1e-8, gtol::Real=1e-10,
        xtol::Real=0.0, xtolr::Real=0.0, showtrace::Union{Bool,Integer}=false) where {P,V,M}
    showtrace === true && (showtrace = 20)
    showtrace === false && (showtrace = 0)
    return NonlinearSystem{P, V, M, typeof(solver)}(fdf, x, fx, dx, solver,
        convert(Int, maxiter), convert(Float64, ftol), convert(Float64, gtol),
        convert(Float64, xtol), convert(Float64, xtolr), normal, inprogress, showtrace)
end

nvar(s::NonlinearSystem) = length(s.fdf.x_f)
nequ(s::NonlinearSystem) = length(s.fdf.F)
size(s::NonlinearSystem) = (nvar(s), nequ(s))
size(s::NonlinearSystem, dim::Integer) =
    dim == 1 ? nequ(s) : dim == 2 ? nvar(s) : throw(ArgumentError("dim can only be 1 or 2"))

@inline _test_ftol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(s.fx[i]) < s.ftol)

@inline _test_gtol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(getgrad(s.solver)[i] * max(s.x[i], 1)) < s.gtol)

@inline _test_xtol_i(s::NonlinearSystem, i::Int) =
    @inbounds(abs(s.dx[i]) < s.xtol + s.xtolr * abs(s.x[i]))

function assess_state(s::NonlinearSystem{P}) where P
    if s.iterstate !== normal
        return failed
    elseif s.solver.iter >= s.maxiter
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

@inline function iterate(s::NonlinearSystem, state=getiter(s))
    solver = s.solver
    s.iterstate = solver(s.fdf, s.x, s.fx, s.dx)
    s.exitstate = assess_state(s)
    # How iter changes depends on the specific algorithm
    return s, getiter(solver) # Termination is never enforced here
end

function solve!(s::NonlinearSystem)
    iter = getiter(s.solver)
    while s.exitstate === inprogress
        s.showtrace > 0 && iszero((iter-1) % s.showtrace) &&
            _show_trace(stdout, s.solver, true)
        _, iter = iterate(s, iter)
    end
    s.showtrace > 0 && !(iszero((iter-1) % s.showtrace)) &&
        _show_trace(stdout, s.solver, true)
    return s
end

init(algo::AbstractAlgorithm, args...; kwargs...) =
    init(typeof(algo), args...; kwargs...)

init(Algo::Type{<:AbstractAlgorithm}, f::Function, x0::AbstractVector; kwargs...) =
    init(Algo, OnceDifferentiable(f, similar(x0), similar(x0)), x0; kwargs...)

init(Algo::Type{<:AbstractAlgorithm}, f::Function, j::Function, x0::AbstractVector;
    kwargs...) =
        init(Algo, OnceDifferentiable(f, j, similar(x0), similar(x0)), x0; kwargs...)

function show(io::IO, s::NonlinearSystem{P}) where P
    print(io, nequ(s), '×', nvar(s), ' ', typeof(s).name.name, '{', P, "}(")
    print(io, algorithmtype(s.solver), ", ", getfnorm(s.solver), ", ", s.exitstate, ')')
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
    println(io, rpad("  Solver exit state: ", w), s.exitstate)
    println(io, rpad("  # Iterations: ", w), getiter(s.solver))
    println(io, rpad("  # Residual calls (f): ", w), s.fdf.f_calls[1])
    print(io, rpad("  # Jacobian calls (df/dx): ", w), s.fdf.df_calls[1])
end
