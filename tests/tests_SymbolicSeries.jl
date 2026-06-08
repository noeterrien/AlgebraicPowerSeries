include("../AlgebraicPowerSeries.jl")

using Test

@variables x y z t
@variables i j k

sin_series = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
K = selfseries_symbols(2,2)

sin_ss = SymbolicSeries(sin_series)
K_ss = SymbolicSeries(K, [0])

getNum((K_ss[1,1] + K_ss[1,2] + K_ss[2,1]) * 4, 2)
getNum(K_ss[1,1] / 3, 1)
getNum(x*K_ss[2,2], 4)


@test getSymbolics(sin_ss*sin_ss, 4) ≈ getSymbolics((sin_ss*sin_ss)[4])

sincos_series = TaylorExpansionSeries{Float64}(:sincos, [x,y], [sin(x), cos(y)], [0,0])
compute_coefficients!(sincos_series, 6)
sincos_ss = SymbolicSeries(sincos_series)
@test getValue(sincos_ss[1][1,1]) ≈ 0
@test getValue(sincos_ss[2][0,2]) ≈ -1/2

sincos_ss[1](x,y)/3
K_ss(x)

for i in 0:3, j in 0:3, k in 0:3
    if (i,j,k) != (1,0,0) && (i,j,k) != (3,0,0)
        @test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[i,j,k]) ≈ 0
    end
end
@test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[1,0,0]) ≈ 1
@test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[3,0,0]) ≈ -2/3