@inline enorm2(x::AbstractArray) = sum(abs2, x)
@inline enorm(x::AbstractArray) = sqrt(enorm2(x))

@inline function set_scale!(diagn::AbstractVector{TF}, J::AbstractMatrix) where TF
    @inbounds for (j, col) in enumerate(eachcol(J))
        sc = enorm(col)
        diagn[j] = ifelse(iszero(sc), one(TF), sc)
    end
    return diagn
end

@inline function update_scale!(diagn::AbstractVector{TF}, J::AbstractMatrix) where TF
    factor = convert(TF, 0.1)
    @inbounds for (j, col) in enumerate(eachcol(J))
        diagn[j] = max(factor * diagn[j], enorm(col))
    end
    return diagn
end

@inline function scaled_enorm(diagn::AbstractVector{TF}, v::AbstractVector{TF}) where TF
    e2 = zero(TF)
    @inbounds for i in eachindex(v)
        e2 += (diagn[i] * v[i])^2
    end
    return sqrt(e2)
end

@inline function _value!!(fdf::OnceDifferentiable, x, dx)
    F = fdf.F
    xdx = fdf.x_f
    xdx .= x .+ dx
    fdf.f(F, xdx)
    fdf.f_calls[1] += 1
    return F
end

default_problemtype(fdf::OnceDifferentiable) =
    length(fdf.x_f) == length(fdf.F) ? RootFinding : LeastSquares
