@testset "luupdate!" begin
    for N in (1, 20)
        A = rand(N, N)
        w = rand(N)
        v = rand(N)
        f = lu(A)
        A1 = A + w * v'
        p = zeros(Int, N)
        f1 = luupdate!(f, p, w, v)
        LU1 = f1.L * f1.U
        PA1 = f.P * A1
        @test PA1 ≈ LU1
    end
end

@testset "DenseLUSolver" begin
    M, N = 5, 5
    J = rand(M, N)
    f = rand(M)
    s = init(default_linsolvertype(J, f, RootFinding), J, f)
    Y = zeros(N)
    @test solve!(s, Y, J, f) ≈ lu(J) \ f
end

@testset "DenseCholeskySolver" begin
    M, N = 5, 3
    J = rand(M, N)
    f = rand(M)
    s = init(default_linsolvertype(J, f, LeastSquares), J, f)
    Y = zeros(N)
    @test solve!(s, Y, J, f) ≈ cholesky(J'J) \ J'f
end
