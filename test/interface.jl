@testset "interface" begin
    x0 = zeros(2)
    p = init(Hybrid, f!, x0)
    @test typeof(p).parameters[1] === RootFinding
    @test size(p) == (2, 2)
    @test size(p, 1) == size(p, 2) == 2
    @test_throws ArgumentError size(p, 3)
    @test sprint(show, p)[1:46] == "2×2 NonlinearSystem{RootFinding}(Hybrid, 3.11"
    solve!(p)
    @test Symbol(getexitstate(p)) === :ftol_reached
    @test Symbol(getiterstate(p)) === :normal
    @test getlinsolver(p.solver) isa DenseLUSolver
    # Type parameters for LU are changed
    if VERSION > v"1.8.0-DEV"
        @test sprint(show, MIME("text/plain"), p)[1:290] == """
            2×2 NonlinearSystem{RootFinding, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}}:
              Problem type:                 Root finding
              Algorithm:                    Hybrid
              Candidate (x):                [1.04"""
    end

    fdf = OnceDifferentiable(f!, j!, x0, similar(x0))
    p1 = init(Hybrid, fdf, x0, xtol=1e-3)
    @test typeof(p1).parameters[1] === RootFinding
    solve!(p1)
    @test Symbol(getexitstate(p1)) === :xtol_reached

    p2 = solve(Hybrid{LeastSquares}(), f!, j!, x0)
    @test Symbol(getexitstate(p2)) === :ftol_reached
    @test sprint(show, p2)[1:46] == "2×2 NonlinearSystem{LeastSquares}(Hybrid, 9.8"
    @test sprint(show, MIME("text/plain"), p2)[1:302] == """
        2×2 NonlinearSystem{LeastSquares, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseCholeskySolver{Float64, Int8, Matrix{Float64}, Vector{Float64}}, Vector{Float64}}}:
          Problem type:                 Least squares
          Algorithm:                    Hybrid
          Candidate (x):                [1.04"""

    @test sprint(show, p.solver) ==
        "HybridSolver(iter 10 => ‖f(x)‖₂ = 9.847e-11 ‖∇f(x)‖₂ = 1.964e-01 ‖Ddx‖₂ = 1.821e-08 δ = 5.647e-06 ρ = 9.960e-01)"
    # Type parameters for LU are changed
    if VERSION > v"1.8.0-DEV"
        @test sprint(show, MIME("text/plain"), p.solver)[1:138] == """
            HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}:
              iter   10  =>  ‖f(x)‖₂ =  9.84"""
    end

    p3 = solve(Hybrid{LeastSquares}, f!, j!, x0, gtol=1)
    @test Symbol(getexitstate(p3)) === :gtol_reached
end
