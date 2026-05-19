using Test
using BenchmarkTools

include("../AlgebraicPowerSeries.jl")

@variables x ξ

∑λᵢ = TaylorSeries{Float64}(:sin, [x], [sin(x)], [0])
compute_coefficients(∑λᵢ, 10)

@variables i j k
@variables Kᵢ₀ Kᵢⱼ Kᵢ₍ⱼ₊₂₎ Kₖⱼ K₍ᵢ₊₂₎ⱼ K₍ᵢ₊₂₎₍ⱼ₊₂₎
@variables λᵢ₋₁ λᵢ₋ₖ
@variables Σⱼ₌₀ⁱKᵢⱼ Bᵢⱼ

sc_Kᵢ₀     = SeriesCoefficient(:self, Kᵢ₀, [i,0], [i], (1,))
sc_Kᵢⱼ     = SeriesCoefficient(:self, Kᵢⱼ, [i,j], [i,j], (1,))
sc_K₍ᵢ₊₂₎ⱼ = SeriesCoefficient(:self, K₍ᵢ₊₂₎ⱼ, [i+2,j], [i,j], (1,))
sc_Kᵢ₍ⱼ₊₂₎ = SeriesCoefficient(:self, Kᵢ₍ⱼ₊₂₎, [i,j+2], [i,j], (1,))
sc_K₍ᵢ₊₂₎₍ⱼ₊₂₎ = SeriesCoefficient(:self, K₍ᵢ₊₂₎₍ⱼ₊₂₎, [i+2,j+2], [i,j], (1,))
sc_Kₖⱼ     = SeriesCoefficient(:self, Kₖⱼ, [k,j], [k,j], (1,))

sc_λᵢ₋₁ = SeriesCoefficient(∑λᵢ, λᵢ₋₁, [i-1], [i], (1,))
sc_λᵢ₋ₖ = SeriesCoefficient(∑λᵢ, λᵢ₋ₖ, [i-k], [i,k], (1,))

R1 = RecurrentRelation(Kᵢ₀ ~ 0, [i], [(0,:∞)], [sc_Kᵢ₀], [])

sum(formulae::Vector) = +(formulae...)
ef_Σⱼ₌₀ⁱKᵢⱼ = ExpandableFormula(:Σⱼ₌₀ⁱKᵢⱼ, Σⱼ₌₀ⁱKᵢⱼ, Kᵢⱼ, [i], [j], [(0,i)], [], [sc_Kᵢⱼ], sum)
println(expand(ef_Σⱼ₌₀ⁱKᵢⱼ, Dict(i=>5)))
R2 = RecurrentRelation(ef_Σⱼ₌₀ⁱKᵢⱼ ~ -λᵢ₋₁/(2*i), [i], [(1,:∞)], [sc_λᵢ₋₁], [ef_Σⱼ₌₀ⁱKᵢⱼ])



ef_Bᵢⱼ = ExpandableFormula(:Bᵢⱼ, Bᵢⱼ, Kₖⱼ*λᵢ₋ₖ, [i,j], [k], [(j,i)], [], [sc_Kₖⱼ,sc_λᵢ₋ₖ], sum)
R3 = RecurrentRelation((i+2-j)*(i+1-j)*K₍ᵢ₊₂₎ⱼ - (j+2)*(j+1)*K₍ᵢ₊₂₎₍ⱼ₊₂₎ ~ Bᵢⱼ, [i,j], 
                       [(0,:∞),(0,i)], [sc_K₍ᵢ₊₂₎ⱼ, sc_K₍ᵢ₊₂₎₍ⱼ₊₂₎], [ef_Bᵢⱼ]);

rs = RecurrentSeries{Float64}(:K, (1,), [x,ξ], [0,0], [R1, R2, R3])

println()