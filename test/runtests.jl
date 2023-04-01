using Test
using NonlinearSystems

using LinearAlgebra
using NonlinearSystems: luupdate!, update!, default_linsolver, getlinsolver,
    getiter, getfnorm, getpnorm

const tests = [
    "linsolve",
    "interface",
    "burkardt_test_nonlin",
    "minpack_test_hybrj"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
