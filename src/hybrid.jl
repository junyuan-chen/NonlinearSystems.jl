"""
    Hybrid{P} <: AbstractAlgorithm{P}

A modified version of Powell's hybrid method (a trust region method with dogleg).
The essentially same algorithm can solve both root-finding problems and
least-squares problem.
To indicate the problem type, set either `RootFinding` or `LeastSquares` as type parameter.
For keyword arguments accepted by `init` and `solve` when using this algorithm,
see [`HybridSolver`](@ref).

# References
- **Moré, Jorge J., Danny C. Sorenson, Burton S. Garbow, and Kenneth E. Hillstrom.** 1984.
  "The MINPACK Project."
  In *Sources and Development of Mathematical Software*,
  ed. Wayne R. Cowell, 88-111. New Jersey: Prentice-Hall.
- **Nocedal, Jorge, and Stephen J. Wright.** 2006.
  *Numerical Optimization.* 2nd ed. New York: Springer.
- **Powell, Michael J. D.** 1970.
  "A Hybrid Method for Nonlinear Equations."
  In *Numerical Methods for Nonlinear Algebraic Equations*,
  ed. Philip Rabinowitz, 87-114. London: Gordon and Breach.
"""
struct Hybrid{P} <: AbstractAlgorithm{P} end

struct HybridSolverState{T}
    iter::Int
    ncfail::Int
    ncsuc::Int
    nslow1::Int
    nslow2::Int
    moved::Bool
    fnorm::T
    pnorm::T
    δ::T
    ρ::T
end

struct HybridSolver{T, L, V} <: AbstractSolver{T}
    state::RefValue{HybridSolverState{T}}
    linsolver::L
    diagn::V
    newton::V
    grad::V
    df::V
    Jdx::V
    w::V
    v::V
    factor_init::T
    factor_up::T
    factor_down::T
    scaling::Bool
    rank1update::Bool
    thres_jac::Int
    thres_nslow1::Int
    thres_nslow2::Int
    warn::Bool
end

"""
    HybridSolver(fdf::OnceDifferentiable, x, fx, J, P; kwargs...)

Construct `HybridSolver` for solving a problem of type `P` with [`Hybrid`](@ref) method.
Users are not expected to call this method directly
but should pass keyword arguments to `init` or `solve` instead.
See also [`Hybrid`](@ref).

# Keywords
- `linsolver=default_linsolver(fdf, x0, P)`: solver for the underlying linear problems.
- `factor_init::Real=1.0`: a factor for scaling the initial trust region radius.
- `factor_up::Real=2.0`: a factor for expanding the trust region radius.
- `factor_down::Real=0.5`: a factor for shrinking the trust region radius.
- `scaling::Bool=true`: allow improving the scaling of trust region.
- `rank1update::Bool=true`: allow using rank-1 update for the Jacobian matrix
  and factorization.
- `thres_jac::Integer=2`: recompute the Jacobian matrix
  if the trust region is shrinked consecutively by the specified number of times;
  setting a non-positive value results in recomputing the Jacobian matrix after each step.
- `thres_nslow1::Integer=10`: signal slow solver progress if the reduction in residual norm
  remains small for the specified number of consecutive steps.
- `thres_nslow2::Integer=5`: signal slow solver progress
  if there is no expansion of trust region after recomputing the Jacobian matrix
  in the specified number of consecutive steps.
- `warn::Bool=true`: print a warning message for slow solver progress
"""
function HybridSolver(fdf::OnceDifferentiable, x::AbstractVector, fx::AbstractVector,
        J::AbstractMatrix, P::Type{<:ProblemType};
        linsolver=default_linsolver(fdf, x0, P),
        factor_init::Real=1.0, factor_up::Real=2.0, factor_down::Real=0.5,
        scaling::Bool=true, rank1update::Bool=true,
        thres_jac::Integer=2, thres_nslow1::Integer=10, thres_nslow2::Integer=5,
        warn::Bool=true)
    M, N = size(J) # Assume the sizes of x, fx and J are all compatible
    P === RootFinding && M != N && throw(DimensionMismatch(
        "the number of variables must match the number of equations in a root-finding problem"))
    diagn = similar(x)
    newton = similar(x)
    grad = similar(x)
    nan = convert(eltype(x), NaN)
    fill!(grad, nan)
    df = similar(fx)
    Jdx = similar(fx)
    w = similar(fx)
    v = similar(x)
    # Initial values should have been evaluated when initializing linsolver
    copyto!(fx, fdf.F)
    scaling ? set_scale!(diagn, J) : fill!(diagn, one(eltype(diagn)))
    fnorm = enorm(fx)
    T = typeof(fnorm)
    factor_init = convert(T, factor_init)
    factor_up = convert(T, factor_up)
    factor_down = convert(T, factor_down)
    factor_init > 0 && factor_up > 0 && factor_down > 0 ||
        throw(ArgumentError("all factors must be positive"))
    Dx = scaled_enorm(diagn, x)
    δ = Dx > 0 ? factor_init * Dx : factor_init
    state = HybridSolverState(1, 0, 0, 0, 0, false, fnorm, nan, δ, nan)
    return HybridSolver(Ref(state), linsolver, diagn,
        newton, grad, df, Jdx, w, v, factor_init, factor_up, factor_down,
        scaling, rank1update, convert(Int, thres_jac),
        convert(Int, thres_nslow1), convert(Int, thres_nslow2), warn)
end

# ! This method reuses arrays from s
function init(s::NonlinearSystem{P,V,M,<:HybridSolver{T}}, x0::V;
        factor_init::Real=s.solver.factor_init, factor_up::Real=s.solver.factor_up,
        factor_down::Real=s.solver.factor_down, scaling::Bool=s.solver.scaling,
        rank1update::Bool=s.solver.rank1update, thres_jac::Integer=s.solver.thres_jac,
        thres_nslow1::Integer=s.solver.thres_nslow1,
        thres_nslow2::Integer=s.solver.thres_nslow2,
        warn::Bool=s.solver.warn,
        initf::Bool=true, initdf::Bool=true, kwargs...) where {P,V,M,T}
    s.x === x0 || copyto!(s.x, x0)
    nan = convert(eltype(s.x), NaN)
    fill!(s.dx, nan)
    ss, fdf = s.solver, s.fdf
    linsolver = getlinsolver(s)
    init(linsolver, fdf, s.x; initf=initf, initdf=initdf)
    fill!(ss.grad, nan)
    copyto!(s.fx, fdf.F)
    diagn = ss.diagn
    scaling ? set_scale!(diagn, fdf.DF) : fill!(diagn, one(eltype(diagn)))
    fnorm = enorm(s.fx)
    factor_init = convert(T, factor_init)
    factor_up = convert(T, factor_up)
    factor_down = convert(T, factor_down)
    factor_init > 0 && factor_up > 0 && factor_down > 0 ||
        throw(ArgumentError("all factors must be positive"))
    Dx = scaled_enorm(diagn, s.x)
    δ = Dx > 0 ? factor_init * Dx : factor_init
    state = HybridSolverState(1, 0, 0, 0, 0, false, fnorm, nan, δ, nan)
    solver = HybridSolver(Ref(state), linsolver, diagn,
        ss.newton, ss.grad, ss.df, ss.Jdx, ss.w, ss.v, factor_init, factor_up, factor_down,
        scaling, rank1update, convert(Int, thres_jac),
        convert(Int, thres_nslow1), convert(Int, thres_nslow2), warn)
    return NonlinearSystem(P, s.fdf, s.x, s.fx, s.dx, solver;
        lower=s.lb, upper=s.ub, maxiter=s.maxiter, ftol=s.ftol, gtol=s.gtol,
        xtol=s.xtol, xtolr=s.xtolr, showtrace=s.showtrace, kwargs...)
end

function dogleg!(dx, linsolver, J, fx, diagn, δ, newton, grad, w)
    # Obtain Newton step by solving the linear problem
    solve!(linsolver, newton, J, fx)
    newton .*= -one(eltype(newton))

    # Newton step is within the trust region?
    qnorm = scaled_enorm(diagn, newton)
    if qnorm < δ
        copyto!(dx, newton)
        return dx
    end

    # Compute gradient
    mul!(grad, J', fx)
    grad ./= .-diagn

    # Is gradient norm too small?
    gnorm = enorm(grad)
    if gnorm < 1e-7
        dx .= (δ / qnorm) .* newton
        return dx
    end

    # Compute the norm of Cauchy point
    grad ./= gnorm .* diagn
    mul!(w, J, grad)
    sgnorm = gnorm / enorm2(w)

    # Cauchy point is out of the trust region
    if sgnorm > δ
        dx .= δ .* grad
        return dx
    end

    # Compute the optimal point on dogleg path
    bnorm = enorm(fx)
    bg = bnorm / gnorm
    bq = bnorm / qnorm
    dq = δ / qnorm
    sd = sgnorm / δ
    sd2 = sd^2
    t = bg * bq * sd
    u = t - dq
    α = dq * (1 - sd2) / (t - dq * sd2 + sqrt(u^2 + (1 - dq^2) * (1 - sd2)))
    β = (1 - α) * sgnorm
    dx .= α .* newton .+ β .* grad
    return dx
end

function (s::HybridSolver{T})(fdf::OnceDifferentiable, x::AbstractVector,
        fx::AbstractVector, dx::AbstractVector, lb, ub) where T
    xtrial, ftrial, J = fdf.x_f, fdf.F, fdf.DF
    p1, p5, p001, p0001 = 0.1, 0.5, 0.001, 0.0001
    st, linsolver, diagn, Jdx = s.state[], s.linsolver, s.diagn, s.Jdx
    iter, ncfail, ncsuc, nslow1, nslow2, moved, fnorm, pnorm, δ, ρ =
        st.iter, st.ncfail, st.ncsuc, st.nslow1, st.nslow2, st.moved,
            st.fnorm, st.pnorm, st.δ, st.ρ

    # Update trust region radius based on effectiveness of the last trial
    if !isnan(ρ)
        if ρ < p1
            δ *= s.factor_down
            ncsuc = 0
            ncfail += 1
        else
            if ρ > 1 - p1
                δ = pnorm * s.factor_up
            elseif (ρ > p5 || ncsuc > 0)
                δ = max(δ, pnorm * s.factor_up)
            end
            ncfail = 0
            ncsuc += 1
        end
    end

    # Recompute Jacobian entirely or use rank-1 update?
    # The former is done at most once per iter
    # The latter is done repetitively to exploit information from df in last trial
    recompute_jac = s.thres_jac > 0 ? ncfail === s.thres_jac : moved
    if recompute_jac
        jacobian!!(fdf, x)
        s.thres_jac > 0 && (nslow2 += 1)
        s.scaling && update_scale!(diagn, J)
        update!(linsolver, J)
    elseif s.rank1update && (pnorm > eps(T) || iter === 1 && ncfail > 0)
        s.w .= (s.df .- Jdx) ./ pnorm
        s.v .= diagn.^2 .* dx ./ pnorm
        # Rank-1 update of Jacobian and factorization
        update!(linsolver, J, s.w, s.v)
        s.scaling && update_scale!(diagn, J)
    end

    # Obtain optimal step given the trust region
    # w is used as a cache
    dogleg!(dx, linsolver, J, fx, diagn, δ, s.newton, s.grad, s.w)

    # Impose box constraints
    if lb !== nothing
        @inbounds for i in eachindex(dx)
            dx[i] = max(dx[i], lb[i] - x[i])
        end
    end
    if ub !== nothing
        @inbounds for i in eachindex(dx)
            dx[i] = min(dx[i], ub[i] - x[i])
        end
    end

    # Compute actual reduction and predicted reduction
    pnorm = scaled_enorm(diagn, dx)
    iter === 1 && δ > pnorm && (δ = pnorm)
    _value!!(fdf, x, dx)
    s.df .= ftrial .- fx
    fnorm1 = enorm(ftrial)
    actred = fnorm1 < fnorm ? 1 - fnorm1 / fnorm : -one(T)
    mul!(Jdx, J, dx)
    fp = s.w # Reuse w as cache
    fp .= Jdx .+ fx
    fnorm1p = enorm(fp)
    prered = fnorm1p < fnorm ? 1 - fnorm1p / fnorm : zero(T)
    ρ = prered > 0 ? actred / prered : zero(T)

    # Step accepted?
    if ρ > p0001
        copyto!(x, xtrial)
        copyto!(fx, ftrial)
        fnorm = fnorm1
        iter += 1
        moved = true
    else
        moved = false
    end

    nslow1 = actred > p001 ? 0 : nslow1 + 1
    actred > p1 && (nslow2 = 0)

    s.state[] =
        HybridSolverState(iter, ncfail, ncsuc, nslow1, nslow2, moved, fnorm, pnorm, δ, ρ)

    if nslow1 === s.thres_nslow1
        if nslow2 >= s.thres_nslow2
            s.warn && @warn "iteration $(iter) is not making progress even with reevaluations of Jacobians; try a smaller value with option thres_jac"
            return iter, jac_noprogress
        else
            s.warn && @warn "iteration $(iter) is not making progress"
            return iter, eval_noprogress
        end
    else
        return iter, normal
    end
end

function init(::Type{Hybrid{P}}, fdf::OnceDifferentiable, x0::AbstractVector;
        linsolver=default_linsolver(fdf, x0, P),
        factor_init::Real=1.0, factor_up::Real=2.0, factor_down::Real=0.5,
        scaling=true, rank1update=true,
        thres_jac=2, thres_nslow1=10, thres_nslow2=5, warn=true, kwargs...) where P
    x = copy(x0)
    fx = copy(fdf.F)
    dx = similar(x) # Preserve the array type
    fill!(dx, convert(eltype(x), NaN))
    solver = HybridSolver(fdf, x, fx, fdf.DF, P; linsolver=linsolver,
        factor_init=factor_init, factor_up=factor_up, factor_down=factor_down,
        scaling=scaling, rank1update=rank1update,
        thres_jac=thres_jac, thres_nslow1=thres_nslow1, thres_nslow2=thres_nslow2, warn=warn)
    # Remaining kwargs are handled by NonlinearSystem constructor
    return NonlinearSystem(P, fdf, x, fx, dx, solver; kwargs...)
end

init(::Type{Hybrid}, fdf::OnceDifferentiable, x0::AbstractVector; kwargs...) =
    init(Hybrid{default_problemtype(fdf)}, fdf, x0; kwargs...)

# Assume the problem is RootFinding if OnceDifferentiable is not constructed
init(::Type{Hybrid}, args...; kwargs...) =
    init(Hybrid{RootFinding}, args...; kwargs...)

algorithmtype(::HybridSolver) = Hybrid

function show(io::IO, s::HybridSolver)
    print(io, typeof(s).name.name)
    print(io, "(iter ", getiter(s), " => ")
    @printf(io, "‖f(x)‖₂ = %9.3e ‖∇f(x)‖₂ = %9.3e ‖Ddx‖₂ = %9.3e δ = %9.3e ρ = %9.3e)",
        getfnorm(s), enorm(getgrad(s)), getpnorm(s), s.state[].δ, s.state[].ρ)
end

function _show_trace(io::IO, s::HybridSolver, newline::Bool)
    print(io, "  iter ", lpad(getiter(s), 4), "  =>  ")
    @printf(io,
        "‖f(x)‖₂ = %13.6e  ‖∇f(x)‖₂ = %13.6e  ‖Ddx‖₂ = %13.6e  δ = %13.6e  ρ = %13.6e",
        getfnorm(s), enorm(getgrad(s)), getpnorm(s), s.state[].δ, s.state[].ρ)
    newline && println(io)
end

show(io::IO, ::MIME"text/plain", s::HybridSolver) =
    (println(io, typeof(s), ':'); _show_trace(io, s, false))
