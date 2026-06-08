using Test

include("../AlgebraicPowerSeries.jl")

@variables K¹¹₍ᵢ₊₂₎ⱼₖₗ
@variables i j k l

@variables x y z t 
multidim_sin_ps = TaylorExpansionSeries{Float64}(:multisin, [x,y,z,t], sin.([x;y;;z;t]), [0,0,0,0])
compute_coefficients!(multidim_sin_ps, 3)
@test getValue(multidim_sin_ps[1,1][1,0,0,0]) ≈ 1
@test getValue(multidim_sin_ps[1,1][3,0,0,0]) ≈ -1/6
@test getValue(multidim_sin_ps[2,1][3,3,0,0]) ≈ -1/6
@test getValue(multidim_sin_ps[2,2][3,3,3,3]) ≈ -1/6


for (c, v) in zip([0, π/2, π, -π/2], [0, 1-π^2/8+π^4/384, π-π^3/6+π^5/120, -1+π^2/8-π^4/384])
    sin_ps = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [c])
    compute_coefficients!(sin_ps, 5)
    built = build(sin_ps, 5)
    @test built(0)[1] ≈ v
end