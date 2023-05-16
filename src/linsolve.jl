const PFac = PositiveFactorizations

function apply_ipiv_perm!(w::AbstractVector, p::AbstractVector{<:Integer})
    @inbounds for i in eachindex(p)
        w[i], w[p[i]] = w[p[i]], w[i]
    end
    return w
end

function luupdate!(lu::LU, w::AbstractVector, v::AbstractVector)
    m, n = size(lu)
    apply_ipiv_perm!(w, getfield(lu, :ipiv))
    F = getfield(lu, :factors)
    @inbounds for i in 1:min(m, n)
        wi = w[i]
        vi = v[i]
        Fii = F[i,i] + wi * vi
        F[i,i] = Fii
        vi = vi / Fii
        v[i] = vi
        for j in i+1:m
            Fji = F[j,i]
            wj = w[j] - wi * Fji
            w[j] = wj
            F[j,i] = Fji + wj * vi
        end
        for j in i+1:n
            vj = v[j]
            Fij = F[i,j] + wi * vj
            F[i,j] = Fij
            v[j] = vj - vi * Fij
        end
    end
    return lu
end

mutable struct DenseLUSolver{F}
    ws::LUWs
    fac::F
end

default_linsolver(fdf::OnceDifferentiable{V, M, V}, x0::V,
    ::Type{RootFinding}) where {V<:AbstractVector, M<:AbstractMatrix} =
        init(DenseLUSolver, fdf, x0)

init(::Type{DenseLUSolver}, fdf::OnceDifferentiable, x0::AbstractVector) =
    (_init!(fdf, x0); init(DenseLUSolver, fdf.DF, fdf.F))

function init(::Type{DenseLUSolver}, J::AbstractMatrix, f::AbstractVector)
    ws = LUWs(J)
    # Do not destroy the original J
    fac = LU(LAPACK.getrf!(ws, copy(J))...)
    return DenseLUSolver(ws, fac)
end

init(s::DenseLUSolver, fdf::OnceDifferentiable, x0::AbstractVector) =
    (_init!(fdf, x0); update!(s, fdf.DF); s)

update!(s::DenseLUSolver{<:LU}, J::AbstractMatrix) =
    (s.fac = LU(LAPACK.getrf!(s.ws, copyto!(getfield(s.fac, :factors), J))...); nothing)

update!(s::DenseLUSolver{<:LU}, J::AbstractMatrix, w::AbstractVector, v::AbstractVector) =
    (BLAS.ger!(one(eltype(J)), w, v, J); luupdate!(s.fac, w, v); nothing)

solve!(s::DenseLUSolver{<:LU}, Y, J, F) = ldiv!(Y, s.fac, F)

# A non-allocating version of ldlt! from PositiveFactorizations.jl
function _ldlt!(A::AbstractMatrix{T}, d::AbstractVector; tol=PFac.default_tol(A),
        blocksize=PFac.default_blocksize(T)) where T
    K = size(A, 1)
    for j = 1:blocksize:K
        jend = min(K, j+blocksize-1)
        B11 = view(A, j:jend, j:jend)
        d1 = view(d, j:jend)
        PFac.solve_diagonal!(B11, d1, tol)
        if jend < K
            B21 = view(A, jend+1:K, j:jend)
            PFac.solve_columns!(B21, d1, B11)
            B22 = view(A, jend+1:K, jend+1:K)
            PFac.update_columns!(B22, d1, B21)
        end
    end
    return Cholesky(A, :L, BLAS.BlasInt(0))
end

# A modified version that does not throw LinearAlgebra.PosDefException(i)
function _lowrankdowndate!(C::Cholesky, v::AbstractVector)
    A = C.factors
    n = length(v)
    size(C, 1) == n || throw(DimensionMismatch(
        "updating vector must fit size of factorization"))
    if C.uplo == 'U'
        conj!(v)
    end
    @inbounds for i = 1:n
        Aii = A[i,i]
        # Compute Givens rotation
        s = conj(v[i]/Aii)
        s2 = abs2(s)
        if s2 > 1
            return true
        end
        c = sqrt(1 - abs2(s))
        # Store new diagonal element
        A[i,i] = c*Aii
        # Update remaining elements in row/column
        if C.uplo == 'U'
            for j = i + 1:n
                vj = v[j]
                Aij = (A[i,j] - s*vj)/c
                A[i,j] = Aij
                v[j] = -s'*Aij + c*vj
            end
        else
            for j = i + 1:n
                vj = v[j]
                Aji = (A[j,i] - s*vj)/c
                A[j,i] = Aji
                v[j] = -s'*Aji + c*vj
            end
        end
    end
    return false
end

function _cholesky!(A::AbstractMatrix, d::AbstractVector)
    fac = cholesky!(Hermitian(A), check=false)
    ispos = iszero(getfield(fac, :info)) # See LinearAlgebra.checkpositivedefinite()
    # _ldlt! always gives a solution but sometimes prevents convergence
    # Only use _ldlt! when cholesky! fails
    return ispos ? fac : _ldlt!(A, d)
end

mutable struct DenseCholeskySolver{TF, TI, M, V, V1<:Union{V,Nothing}}
    JtJ::M
    Jtf::V
    v1::V1
    v2::V1
    d::Vector{TI}
    fac::Cholesky{TF, M}
end

default_linsolver(fdf::OnceDifferentiable{V, M, V}, x0::V,
    ::Type{LeastSquares}) where {V<:AbstractVector, M<:AbstractMatrix} =
        init(DenseCholeskySolver, fdf, x0)

init(::Type{DenseCholeskySolver}, fdf::OnceDifferentiable, x0::AbstractVector; kwargs...) =
    (_init!(fdf, x0); init(DenseCholeskySolver, fdf.DF, fdf.F; kwargs...))

function init(::Type{DenseCholeskySolver}, J::AbstractMatrix, f::AbstractVector;
        rank1chol::Bool=length(f)>3)
    JtJ = J'J
    Jtf = J'f
    d = Vector{PFac.signtype(eltype(J))}(undef, size(J, 1))
    if rank1chol
        v1, v2 = similar(Jtf), similar(Jtf)
        return DenseCholeskySolver(JtJ, Jtf, v1, v2, d, _cholesky!(JtJ, d))
    else
        return DenseCholeskySolver(JtJ, Jtf, nothing, nothing, d, _cholesky!(JtJ, d))
    end
end

init(s::DenseCholeskySolver, fdf::OnceDifferentiable{V, M, V}, x0::V) where {V, M} =
    (_init!(fdf, x0); update!(s, fdf.DF); s)

update!(s::DenseCholeskySolver, J::AbstractMatrix) =
    (s.fac = _cholesky!(mul!(s.JtJ, J', J), s.d); return nothing)

# rank-1 update of factorization may succeed in cases where cholesky! fails
function update!(s::DenseCholeskySolver{TF,TI,M,V,V}, J::AbstractMatrix, w::AbstractVector,
        v::AbstractVector) where {TF,TI,M,V}
    sc2 = enorm2(w) - one(TF) # Save the value before w gets destroyed
    Jtw = s.v1
    mul!(Jtw, J', w)
    BLAS.ger!(one(TF), w, v, J)
    Jtw_v = s.v2 # Need a second cache as lowrankupdate! destroys all inputs
    Jtw_v .= Jtw .+ v
    # Must do all update before downdate to keep fac being positive definite
    lowrankupdate!(s.fac, Jtw_v)
    if sc2 > eps(TF)
        v .*= sqrt(sc2)
        lowrankupdate!(s.fac, v)
    end
    # In case updated J'J is not positive definite, do corrected ldlt instead
    _lowrankdowndate!(s.fac, Jtw) && _ldlt!(mul!(s.JtJ, J', J), s.d)
    if sc2 < -eps(TF)
        v .*= sqrt(-sc2)
        _lowrankdowndate!(s.fac, v) && _ldlt!(mul!(s.JtJ, J', J), s.d)
    end
    return nothing
end

update!(s::DenseCholeskySolver{TF,TI,M,V,Nothing}, J::AbstractMatrix,
    w::AbstractVector, v::AbstractVector) where {TF,TI,M,V} =
        (J .+= w .* v'; update!(s, J); nothing)

solve!(s::DenseCholeskySolver, Y, J, F) =
    ldiv!(Y, s.fac, mul!(s.Jtf, J', F))
