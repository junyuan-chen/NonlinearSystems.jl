const PFac = PositiveFactorizations

# A non-allocating version of ipiv2perm from LinearAlgebra/src/lu.jl
function ipiv2perm!(p::AbstractVector{T}, v::AbstractVector{T}, maxi::Integer) where T
    length(p) == maxi || throw(DimensionMismatch("length of p must be $maxi"))
    @inbounds for i in 1:maxi
        p[i] = i
    end
    @inbounds for i in eachindex(v)
        p[i], p[v[i]] = p[v[i]], p[i]
    end
    return p
end

function luupdate!(lu::LU, p::AbstractVector, w::AbstractVector, v::AbstractVector)
    m, n = size(lu)
    ipiv2perm!(p, getfield(lu, :ipiv), m)
    permute!!(w, p)
    F = getfield(lu, :factors)
    @inbounds for i in 1:m
        F[i,i] += w[i] * v[i]
        v[i] /= F[i,i]
        for j in i+1:n
            w[j] -= w[i] * F[j,i]
            F[j,i] += w[j] * v[i]
        end
        for j in i+1:m
            F[i,j] += w[i] * v[j]
            v[j] -= v[i] * F[i,j]
        end
    end
    return lu
end

mutable struct DenseLUSolver{F}
    ws::LUWs
    fac::F
    perm::Vector{Int}
end

default_linsolvertype(::StridedMatrix, ::StridedVector, ::Type{RootFinding}) =
    DenseLUSolver

function init(::Type{DenseLUSolver}, J::StridedMatrix, f::StridedVector)
    ws = LUWs(J)
    # Do not destroy the original J
    fac = LU(LAPACK.getrf!(ws, copy(J))...)
    return DenseLUSolver(ws, fac, Vector{Int}(undef, size(J, 1)))
end

update!(s::DenseLUSolver{<:LU}, J::AbstractMatrix) =
    LU(LAPACK.getrf!(s.ws, copyto!(getfield(s.fac, :factors), J))...)

update!(s::DenseLUSolver{<:LU}, J::AbstractMatrix, w::AbstractVector, v::AbstractVector) =
    luupdate!(s.fac, s.perm, w, v)

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

mutable struct DenseCholeskySolver{TF, TI, M, V}
    JtJ::M
    Jtf::V
    d::Vector{TI}
    fac::Cholesky{TF, M}
end

default_linsolvertype(::StridedMatrix, ::StridedVector, ::Type{LeastSquares}) =
    DenseCholeskySolver

function _cholesky!(A::AbstractMatrix, d::AbstractVector)
    fac = cholesky!(Hermitian(A), check=false)
    ispos = iszero(getfield(fac, :info)) # See LinearAlgebra.checkpositivedefinite()
    # _ldlt! always gives a solution but sometimes prevents convergence
    # Only use _ldlt! when cholesky! fails
    return ispos ? fac : _ldlt!(A, d)
end

function init(::Type{DenseCholeskySolver}, J::StridedMatrix, f::StridedVector)
    JtJ = J'J
    d = Vector{PFac.signtype(eltype(J))}(undef, size(J, 1))
    return DenseCholeskySolver(JtJ, J'f, d, _cholesky!(JtJ, d))
end

update!(s::DenseCholeskySolver, J::AbstractMatrix) =
    _cholesky!(mul!(s.JtJ, J', J), s.d)

update!(s::DenseCholeskySolver, J::AbstractMatrix, w::AbstractVector, v::AbstractVector) =
    update!(s, J)

solve!(s::DenseCholeskySolver, Y, J, F) =
    ldiv!(Y, s.fac, mul!(s.Jtf, J', F))
