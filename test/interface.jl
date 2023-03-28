@testset "interface" begin
    function f!(F, x)
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)
        return F
    end

    function j!(J, x)
        J[1, 1] = x[2]^3-7
        J[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        J[2, 1] = x[2]*u
        J[2, 2] = u
        return J
    end

    x0 = zeros(2)
    p = init(Hybrid, f!, x0)
    @test typeof(p).parameters[1] === RootFinding
    @test size(p) == (2, 2)
    @test size(p, 1) == size(p, 2) == 2
    @test_throws ArgumentError size(p, 3)
    @test sprint(show, p)[1:46] == "2×2 NonlinearSystem{RootFinding}(Hybrid, 3.11"
    solve!(p)
    @test Symbol(p.exitstate) === :ftol_reached
    @test getlinsolver(p.solver) isa DenseLUSolver
    @test sprint(show, MIME("text/plain"), p)[1:290] == """
        2×2 NonlinearSystem{RootFinding, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}}:
          Problem type:                 Root finding
          Algorithm:                    Hybrid
          Candidate (x):                [1.04"""

    p1 = solve(Hybrid{LeastSquares}(), f!, j!, x0)
    @test Symbol(p1.exitstate) === :ftol_reached
    @test sprint(show, p1)[1:46] == "2×2 NonlinearSystem{LeastSquares}(Hybrid, 9.8"
    @test sprint(show, MIME("text/plain"), p1)[1:302] == """
        2×2 NonlinearSystem{LeastSquares, Vector{Float64}, Matrix{Float64}, HybridSolver{Float64, DenseCholeskySolver{Float64, Int8, Matrix{Float64}, Vector{Float64}}, Vector{Float64}}}:
          Problem type:                 Least squares
          Algorithm:                    Hybrid
          Candidate (x):                [1.04"""

    @test sprint(show, p.solver) ==
        "HybridSolver(iter 10 => ‖f(x)‖₂ = 9.847e-11 ‖∇f(x)‖₂ = 1.964e-01 ‖Ddx‖₂ = 1.821e-08 δ = 5.647e-06 ρ = 9.960e-01)"
    @test sprint(show, MIME("text/plain"), p.solver)[1:138] == """
        HybridSolver{Float64, DenseLUSolver{LU{Float64, Matrix{Float64}, Vector{Int64}}}, Vector{Float64}}:
          iter   10  =>  ‖f(x)‖₂ =  9.84"""
end
