using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Create sample problems
    function p1_f!(out, x)
        n = length(x)
        out[1] = 1.0 - x[1]
        out[2:n] .= 10.0 .* (view(x,2:n) .- view(x,1:n-1) .* view(x, 1:n-1))
        out
    end
    x0_1 = ones(10); x0_1[1] = -1.2
    fdf = OnceDifferentiable(p1_f!, x0_1, similar(x0_1))
    function p15_f!(out, x)
        out[1] = (x[1] * x[1] + x[2] * x[3]) - 0.0001
        out[2] = (x[1] * x[2] + x[2] * x[4]) - 1.0
        out[3] = (x[3] * x[1] + x[4] * x[3]) - 0.0
        out[4] = (x[3] * x[2] + x[4] * x[4]) - 0.0001
        out
    end
    x0_15 = [1.0, 0.0, 0.0, 1.0]
    @compile_workload begin
        solve(Hybrid, p1_f!, x0_1)
        solve(Hybrid, p15_f!, x0_15, thres_jac=0)
        solve(Hybrid{LeastSquares}, fdf, x0_1;
            linsolver=init(DenseCholeskySolver, fdf, x0_1; rank1update=Val(true)))
        solve(Hybrid{LeastSquares}, p15_f!, x0_15, thres_jac=0)
    end
end
