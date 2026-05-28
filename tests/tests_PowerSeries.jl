using Test

include("../AlgebraicPowerSeries.jl")

@variables K¹¹₍ᵢ₊₂₎ⱼₖₗ
@variables i j k l

@variables x y z t 
multidim_sin_ps = TaylorExpansionSeries{Float64}(:multisin, [x,y,z,t], sin.([x;y;;z;t]), [0,0,0,0])
compute_coefficients!(multidim_sin_ps, 3)
@test multidim_sin_ps[1,1,1,0,0,0] ≈ 1
@test multidim_sin_ps[1,1,3,0,0,0] ≈ -1/6
@test multidim_sin_ps[2,1,3,3,0,0] ≈ -1/6
@test multidim_sin_ps[2,2,3,3,3,3] ≈ -1/6

sc_K¹¹₍ᵢ₊₂₎ⱼₖₗ = SeriesCoefficient(multidim_sin_ps, K¹¹₍ᵢ₊₂₎ⱼₖₗ, [i+2,j,k,l], [i,j,k,l], (1,1))
@test getValue(sc_K¹¹₍ᵢ₊₂₎ⱼₖₗ, [1,0,0,0]) ≈ -1/6