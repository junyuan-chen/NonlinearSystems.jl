@testset "interface" begin
    x0 = zeros(2)
    p = init(Hybrid, f!, x0)
    fnorm0 = getfnorm(p.solver)
    pnorm0 = getpnorm(p.solver)
    F0 = copy(p.fdf.F)
    J0 = copy(p.fdf.DF)
    @test typeof(p).parameters[1] === RootFinding
    @test size(p) == (2, 2)
    @test size(p, 1) == size(p, 2) == 2
    @test_throws ArgumentError size(p, 3)
    st1 = sprint(show, p)[1:46]
    @test st1 == "2×2 NonlinearSystem{RootFinding}(Hybrid, 3.11"
    solve!(p)
    x = copy(p.x)
    @test Symbol(getexitstate(p)) === :ftol_reached
    @test Symbol(getiterstate(p)) === :normal
    @test getlinsolver(p.solver) isa DenseLUSolver
    # Type parameters for LU are changed
    if VERSION > v"1.8.0-DEV"
        @test sprint(show, MIME("text/plain"), p)[1:308] == """
            2×2 NonlinearSystem{RootFinding, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}, Nothing, Nothing}:
              Problem type:                 Root finding
              Algorithm:                    Hybrid
              Candidate (x):                [1.04"""
    end

    p0 = init(p, x0; ftol=1e-4)
    @test p0.x == x0
    @test getfnorm(p0.solver) ≈ fnorm0
    @test isequal(getpnorm(p0.solver), pnorm0)
    @test p0.fdf.DF ≈ J0
    @test p0.fdf.f_calls[1] == p0.fdf.df_calls[1] == 1
    @test p0.ftol == 1e-4
    @test sprint(show, p0)[1:46] == st1
    p0 = solve!(p0, x0; ftol=1e-8) # The returned p0 is not the input p0
    @test p0.x ≈ x

    pb = solve(Hybrid, f!, x0, lower=[-1,-1], upper=fill(1.1,2))
    @test pb.x ≈ [0, 1] atol=1e-8
    pb = solve(Hybrid, f!, x0, upper=[0.1,0.5])
    @test pb.x[2] <= 0.5
    @test Symbol(getexitstate(pb)) === :failed
    pb = solve(Hybrid, f!, [1.0, 1.0], lower=[0.1,0.5], upper=fill(2.0,2))
    @test pb.x ≈ [1.1011684360748017, 1.3770065499713244] atol=1e-8
    @test Symbol(getexitstate(pb)) === :ftol_reached
    @test_throws ArgumentError solve(Hybrid, f!, x0, upper=[-1,3])
    @test_throws ArgumentError solve(Hybrid, f!, x0, lower=[0.1,0])

    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    p1 = init(Hybrid, fdf, x0, xtol=1e-3)
    @test typeof(p1).parameters[1] === RootFinding
    solve!(p1)
    @test Symbol(getexitstate(p1)) === :xtol_reached

    p2 = solve(Hybrid{LeastSquares}(), f!, j!, x0)
    @test Symbol(getexitstate(p2)) === :ftol_reached
    @test sprint(show, p2)[1:46] == "2×2 NonlinearSystem{LeastSquares}(Hybrid, 9.8"
    @test sprint(show, MIME("text/plain"), p2)[1:329] == """
        2×2 NonlinearSystem{LeastSquares, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseCholeskySolver{Float64, Int8, Matrix{Float64}, Vector{Float64}, Nothing}, Vector{Float64}}, Nothing, Nothing}:
          Problem type:                 Least squares
          Algorithm:                    Hybrid
          Candidate (x):                [1.04"""

    @test sprint(show, p0.solver) ==
        "HybridSolver(iter 10 => ‖f(x)‖₂ = 9.847e-11 ‖∇f(x)‖₂ = 1.964e-01 ‖Ddx‖₂ = 1.821e-08 δ = 5.647e-06 ρ = 9.960e-01)"
    # Type parameters for LU are changed
    if VERSION > v"1.8.0-DEV"
        @test sprint(show, MIME("text/plain"), p.solver)[1:138] == """
            HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}:
              iter   10  =>  ‖f(x)‖₂ =  9.84"""
    end

    pb = solve(Hybrid{LeastSquares}, f!, [1.0,1.0], lower=[0.1,0.5], upper=fill(2.0,2))
    @test pb.x ≈ [1.1011684360748017, 1.3770065499713244] atol=1e-8

    p3 = solve(Hybrid{LeastSquares}, f!, j!, x0, gtol=1)
    @test Symbol(getexitstate(p3)) === :gtol_reached
    p4 = solve(Hybrid{LeastSquares}, f!, j!, x0, maxiter=2)
    @test Symbol(getexitstate(p4)) === :maxiter_reached

    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    p5 = solve(Hybrid{LeastSquares}, fdf, x0,
        linsolver=init(DenseCholeskySolver, fdf, x0))
    @test Symbol(getexitstate(p5)) === :ftol_reached
    p5 = init(p5, x0, xtol=1e-3)
    solve!(p5)
    @test Symbol(getexitstate(p5)) === :xtol_reached
end
