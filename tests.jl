using Test

include("AlgebraicPowerSeries.jl")

vars = @variables x y z
func = sin(x+y+z)

ps = TaylorSeries{Float64}(vars, func, [0,0,0])
compute_coefficients(ps, 3)