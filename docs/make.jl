using Documenter
using NonlinearSystems

makedocs(
    modules = [NonlinearSystems],
    format = Documenter.HTML(
        canonical = "https://junyuan-chen.github.io/NonlinearSystems.jl/stable/",
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main"
    ),
    sitename = "NonlinearSystems.jl",
    authors = "Junyuan Chen",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "man/getting-started.md",
            "Solver Options" => "man/solver-options.md"
        ]
    ],
    workdir = joinpath(@__DIR__, "..")
)

deploydocs(
    repo = "github.com/junyuan-chen/NonlinearSystems.jl.git",
    devbranch = "main"
)
