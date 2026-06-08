include("../AlgebraicPowerSeries.jl")

using Test

@variables x y
@variables i j k

sin_series = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
K = selfseries_symbols(2,2)

sin_ss = SymbolicSeries(sin_series)
K_ss = SymbolicSeries(K, [0])

(K_ss[1,1] + K_ss[1,2] + K_ss[2,1]) * 4
K_ss[1,1] / 3
x*K_ss[2,2]

@test getSymbolics(sin_ss*sin_ss, 4) ≈ getSymbolics((sin_ss*sin_ss)[4])

sincos_series = TaylorExpansionSeries{Float64}(:sincos, [x,y], [sin(x), cos(y)], [0,0])
compute_coefficients!(sincos_series, 5)
sincos_ss = SymbolicSeries(sincos_series)
@test getValue(sincos_ss[1][1,1]) ≈ 0
@test getValue(sincos_ss[2][0,2]) ≈ -1/2