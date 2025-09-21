function f!(F, x)
    F[1] = (x[1] + 3) * (x[2]^3 - 7) + 18
    F[2] = sin(x[2] * exp(x[1]) - 1)
    return F
end

function j!(J, x)
    J[1,1] = x[2]^3 - 7
    J[1,2] = 3 * x[2]^2 * (x[1] + 3)
    u = exp(x[1]) * cos(x[2] * exp(x[1]) - 1)
    J[2,1] = x[2] * u
    J[2,2] = u
    return J
end

@testset "luupdate!" begin
    for N in (1, 20)
        A = rand(N, N)
        w = rand(N)
        v = rand(N)
        f = lu(A)
        A1 = A + w * v'
        f1 = luupdate!(f, w, v)
        LU1 = f1.L * f1.U
        PA1 = f.P * A1
        @test PA1 ≈ LU1
    end
end

@testset "DenseLUSolver" begin
    x0 = zeros(2)
    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    s = default_linsolver(fdf, x0, RootFinding)
    Y = zeros(2)
    @test solve!(s, Y, copy(fdf.DF), copy(fdf.F)) ≈ fdf.DF \ fdf.F

    x = rand(2)
    init(s, fdf, x)
    F = zeros(2)
    J = zeros(2, 2)
    @test fdf.F == f!(F, x)
    @test fdf.DF == j!(J, x)
    x1 = rand(2)
    init(s, fdf, x1; initf=false)
    @test fdf.F == f!(F, x)
    @test fdf.DF == j!(J, x1)
    x2 = rand(2)
    init(s, fdf, x2; initdf=false)
    @test fdf.F == f!(F, x2)
    @test fdf.DF == j!(J, x1)
end

@testset "DenseCholeskySolver" begin
    x0 = zeros(2)
    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    s = default_linsolver(fdf, x0, LeastSquares; rank1update=Val(false))
    @test typeof(s).parameters[5] === Nothing
    Y = zeros(2)
    J = fdf.DF
    @test solve!(s, Y, copy(J), copy(fdf.F)) ≈ cholesky(J'J) \ J'fdf.F

    w = randn(2)
    v = randn(2)
    J1 = J + w*v'
    JJ1 = J1'J1
    update!(s, J, w, v)
    @test JJ1 ≈ s.fac.U's.fac.U

    M, N = 10, 4
    J = randn(M, N)
    s1 = init(DenseCholeskySolver, J, randn(M))
    @test typeof(s1).parameters[5] === Vector{Float64}
    w = randn(M)
    v = randn(N)
    update!(s1, copy(J), copy(w), copy(v))
    J1 = J .+ w .* v'
    @test s1.fac.U's1.fac.U ≈ J1'J1

    x = rand(2)
    init(s, fdf, x)
    F = zeros(2)
    J = zeros(2, 2)
    @test fdf.F == f!(F, x)
    @test fdf.DF == j!(J, x)
    x1 = rand(2)
    init(s, fdf, x1; initf=false)
    @test fdf.F == f!(F, x)
    @test fdf.DF == j!(J, x1)
    x2 = rand(2)
    init(s, fdf, x2; initdf=false)
    @test fdf.F == f!(F, x2)
    @test fdf.DF == j!(J, x1)
end
