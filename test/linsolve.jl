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
        p = zeros(Int, N)
        f1 = luupdate!(f, p, w, v)
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
end

@testset "DenseCholeskySolver" begin
    x0 = zeros(2)
    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    s = default_linsolver(fdf, x0, LeastSquares)
    Y = zeros(2)
    J = fdf.DF
    @test solve!(s, Y, copy(J), copy(fdf.F)) ≈ cholesky(J'J) \ J'fdf.F
end
