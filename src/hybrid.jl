struct Hybrid{P} <: AbstractAlgorithm{P} end

mutable struct HybridSolver{T, L, V} <: AbstractSolver{T}
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
end

function HybridSolver(fdf::OnceDifferentiable, x::AbstractVector, fx::AbstractVector,
        J::AbstractMatrix, P::Type{<:ProblemType};
        linsolvertype=default_linsolvertype(J, fx, P),
        factor_init::Real=1.0, factor_up::Real=2.0, factor_down::Real=0.5,
        scaling::Bool=true, rank1update::Bool=true,
        thres_jac::Integer=2, thres_nslow1::Integer=10, thres_nslow2::Integer=5)
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
    # Set initial values
    value_jacobian!!(fdf, x)
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
    linsolver = init(linsolvertype, J, fx)
    return HybridSolver(1, 0, 0, 0, 0, false, fnorm, nan, δ, nan, linsolver, diagn,
        newton, grad, df, Jdx, w, v, factor_init, factor_up, factor_down,
        scaling, rank1update, convert(Int, thres_jac),
        convert(Int, thres_nslow1), convert(Int, thres_nslow2))
end

function dogleg!(dx, linsolver, J, fx, diagn, δ, newton, grad)
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
    mul!(dx, J, grad)
    sgnorm = gnorm / enorm2(dx)

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

function (s::HybridSolver)(fdf::OnceDifferentiable, x::AbstractVector,
        fx::AbstractVector, dx::AbstractVector)
    T = eltype(x)
    xtrial, ftrial, J = fdf.x_f, fdf.F, fdf.DF
    p1, p5, p001, p0001 = 0.1, 0.5, 0.001, 0.0001
    linsolver, diagn, Jdx, pnorm = s.linsolver, s.diagn, s.Jdx, s.pnorm

    # Update trust region radius based on effectiveness of the last trial
    ratio = s.ρ
    if !isnan(ratio)
        if ratio < p1
            s.δ *= s.factor_down
            s.ncsuc = 0
            s.ncfail += 1
        else
            if ratio > 1 - p1
                s.δ = pnorm * s.factor_up
            elseif (ratio > p5 || s.ncsuc > 0)
                s.δ = max(s.δ, pnorm * s.factor_up)
            end
            s.ncfail = 0
            s.ncsuc += 1
        end
    end

    # Recompute Jacobian entirely or use rank-1 update?
    # The former is done at most once per iter
    # The latter is done repetitively to exploit information from df in last trial
    recompute_jac = s.thres_jac > 0 ? s.ncfail === s.thres_jac : s.moved
    if recompute_jac
        jacobian!!(fdf, x)
        s.thres_jac > 0 && (s.nslow2 += 1)
        s.scaling && update_scale!(diagn, J)
        update!(linsolver, J)
    elseif s.rank1update && (pnorm > eps(T) || s.iter === 1 && s.ncfail > 0)
        s.w .= (s.df .- Jdx) ./ pnorm
        s.v .= diagn.^2 .* dx ./ pnorm
        # Rank-1 update of Jacobian
        BLAS.ger!(one(T), s.w, s.v, J)
        s.scaling && update_scale!(diagn, J)
        # Rank-1 update of factorization
        update!(linsolver, J, s.w, s.v)
    end

    # Obtain optimal step given the trust region
    dogleg!(dx, linsolver, J, fx, diagn, s.δ, s.newton, s.grad)

    # Compute actual reduction and predicted reduction
    s.pnorm = pnorm = scaled_enorm(diagn, dx)
    s.iter === 1 && s.δ > pnorm && (s.δ = pnorm)
    _value!!(fdf, x, dx)
    s.df .= ftrial .- fx
    fnorm1 = enorm(ftrial)
    fnorm = s.fnorm
    actred = fnorm1 < fnorm ? 1 - fnorm1 / fnorm : -one(T)
    Jdx = s.Jdx
    mul!(Jdx, J, dx)
    fp = s.w # Reuse w as cache
    fp .= Jdx .+ fx
    fnorm1p = enorm(fp)
    prered = fnorm1p < fnorm ? 1 - fnorm1p / fnorm : zero(T)
    s.ρ = ratio = prered > 0 ? actred / prered : zero(T)

    # Step accepted?
    if ratio > p0001
        copyto!(x, xtrial)
        copyto!(fx, ftrial)
        s.fnorm = fnorm1
        s.iter += 1
        s.moved = true
    else
        s.moved = false
    end

    s.nslow1 = actred > p001 ? 0 : s.nslow1 + 1
    actred > p1 && (s.nslow2 = 0)

    if s.nslow1 === s.thres_nslow1
        if s.nslow2 >= s.thres_nslow2
            @warn "iteration $(s.iter) is not making progress even with reevaluations of Jacobians; try a smaller value with option thres_jac"
            return jac_noprogress
        else
            @warn "iteration $(s.iter) is not making progress"
            return eval_noprogress
        end
    else
        return normal
    end
end

function init(::Type{Hybrid{P}}, fdf::OnceDifferentiable, x0::AbstractVector;
        linsolvertype=default_linsolvertype(fdf.DF, fdf.F, P),
        factor_init::Real=1.0, factor_up::Real=2.0, factor_down::Real=0.5,
        scaling=true, rank1update=true,
        thres_jac=2, thres_nslow1=10, thres_nslow2=5, kwargs...) where P
    x = copy(x0)
    fx = copy(fdf.F)
    dx = similar(x) # Preserve the array type
    fill!(dx, convert(eltype(x), NaN))
    solver = HybridSolver(fdf, x, fx, fdf.DF, P; linsolvertype=linsolvertype,
        factor_init=factor_init, factor_up=factor_up, factor_down=factor_down,
        scaling=scaling, rank1update=rank1update,
        thres_jac=thres_jac, thres_nslow1=thres_nslow1, thres_nslow2=thres_nslow2)
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
    print(io, "(iter ", s.iter, " => ")
    @printf(io, "‖f(x)‖₂ = %9.3e ‖∇f(x)‖₂ = %9.3e ‖Ddx‖₂ = %9.3e δ = %9.3e ρ = %9.3e)",
        s.fnorm, enorm(getgrad(s)), s.pnorm, s.δ, s.ρ)
end

function _show_trace(io::IO, s::HybridSolver, newline::Bool)
    print(io, "  iter ", lpad(s.iter, 4), "  =>  ")
    @printf(io,
        "‖f(x)‖₂ = %13.6e  ‖∇f(x)‖₂ = %13.6e  ‖Ddx‖₂ = %13.6e  δ = %13.6e  ρ = %13.6e",
        s.fnorm, enorm(getgrad(s)), s.pnorm, s.δ, s.ρ)
    newline && println(io)
end

show(io::IO, ::MIME"text/plain", s::HybridSolver) =
    (println(io, typeof(s), ':'); _show_trace(io, s, false))
