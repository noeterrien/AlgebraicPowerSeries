using Test
using BenchmarkTools

include("../AlgebraicPowerSeries.jl")

@variables x y z t

sin_ps = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
compute_coefficients!(sin_ps, 5)
@test sin_ps[1] ≈ [0,1,0,-1/6,0,1/120]

sincos_ps = TaylorExpansionSeries{Float64}(:sincos, [x,y], [sin(x)cos(y)], [0,0])
compute_coefficients!(sincos_ps, 3)
@test sincos_ps[1] ≈ [0,1,0,0,0,0,-1/6,0,-1/2,0]

multidim_sin_ps = TaylorExpansionSeries{Float64}(:mutlidim_sin, [x,y,z,t], sin.([x;y;;z;t]), [0,0,0,0])
compute_coefficients!(multidim_sin_ps, 3)
multidim_sin_ps_res = Matrix{Vector{Float64}}(undef, 2, 2)
multidim_sin_ps_res[1,1] = [0,
                            1,0,0,0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            -1/6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

multidim_sin_ps_res[2,1] = [0,
                            0, 1,0,0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/6, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                            
multidim_sin_ps_res[1,2] = [0,
                            0, 0, 1,0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/6, 0, 0, 0]

multidim_sin_ps_res[2,2] = [0,
                            0, 0, 0, 1,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/6]
@test multidim_sin_ps.coefficients ≈ multidim_sin_ps_res

multidim_sin_cos_ps = TaylorExpansionSeries{ComplexF64}(:multidim_sin_cos, [x,y], [sin(x+y),cos(x+y)], [0,0])
compute_coefficients!(multidim_sin_cos_ps, 5)
multidim_sin_cos_res = Vector{Float64}[]
push!(multidim_sin_cos_res, [0, 1, 1,  0,  0,  0, -1/6, -1/2, -1/2, -1/6, 0, 0, 0, 0, 0, 1/120, 1/24, 1/12, 1/12, 1/24, 1/120])
push!(multidim_sin_cos_res, [1, 0, 0, -1/2, -1, -1/2,  0,  0,  0,  0, 1/24, 1/6, 1/4, 1/6, 1/24, 0, 0, 0, 0, 0, 0])
@test multidim_sin_cos_ps.coefficients ≈ multidim_sin_cos_res

sin_halfπ = TaylorExpansionSeries{Float64}(:sin_halfπ, [x], [sin(x)], [π/2])
compute_coefficients!(sin_halfπ, 5)
@test sin_halfπ.coefficients[1] ≈ [1, 0, -1/2, 0, 1/24, 0]

print("Time needed to compute coefficients of a 2x2 sin*cos matrix of 4 variables up to order 20 : ")
@btime begin
multidim_sintimescos_ps = TaylorExpansionSeries{Float64}(:multidim_sintimescos, [x,y,z,t], sin.([x;y;;z;t]).*cos(t), [0,0,0,0])
compute_coefficients!(multidim_sintimescos_ps, 20)
end

darctan_ps = TaylorExpansionSeries{Float64}(:darctan, [x], [1/(x^2+1)], [0])
@profview compute_coefficients!(darctan_ps, 20)