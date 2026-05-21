using Test
using BenchmarkTools

include("../AlgebraicPowerSeries.jl")

@variables x y

#----------------------------------1-D reaction diffusion equation with space-varying reaction------------------------------------

∑λᵢ = TaylorSeries{Float64}(:sin, [x], [sin(x)], [0])
compute_coefficients(∑λᵢ, 10)

@variables i j k
@variables Kᵢ₀ Kᵢⱼ Kᵢ₍ⱼ₊₂₎ Kₖⱼ
@variables λᵢ₋₁ λᵢ₋₂₋ₖ
@variables Σⱼ₌₀ⁱKᵢⱼ B₍ᵢ₋₂₎ⱼ

sc_Kᵢ₀     = SeriesCoefficient(:self, Kᵢ₀, [i,0], [i], (1,))
sc_Kᵢⱼ     = SeriesCoefficient(:self, Kᵢⱼ, [i,j], [i,j], (1,))
sc_Kᵢ₍ⱼ₊₂₎ = SeriesCoefficient(:self, Kᵢ₍ⱼ₊₂₎, [i,j+2], [i,j], (1,))
sc_Kₖⱼ     = SeriesCoefficient(:self, Kₖⱼ, [k,j], [k,j], (1,))

sc_λᵢ₋₁ = SeriesCoefficient(∑λᵢ, λᵢ₋₁, [i-1], [i], (1,))
sc_λᵢ₋₂₋ₖ = SeriesCoefficient(∑λᵢ, λᵢ₋₂₋ₖ, [i-k], [i,k], (1,))

R1 = RecurrentRelation(Kᵢ₀ ~ 0, [i], [(0,:∞)], [sc_Kᵢ₀], [])

sum(formulae::Vector) = +(formulae...)
ef_Σⱼ₌₀ⁱKᵢⱼ = ExpandableFormula(:Σⱼ₌₀ⁱKᵢⱼ, Σⱼ₌₀ⁱKᵢⱼ, Kᵢⱼ, [i], [j], [(0,i)], [], [sc_Kᵢⱼ], sum)
R2 = RecurrentRelation(Σⱼ₌₀ⁱKᵢⱼ ~ -λᵢ₋₁/(2*i), [i], [(1,:∞)], [sc_λᵢ₋₁], [ef_Σⱼ₌₀ⁱKᵢⱼ])



ef_B₍ᵢ₋₂₎ⱼ = ExpandableFormula(:B₍ᵢ₋₂₎ⱼ , B₍ᵢ₋₂₎ⱼ, Kₖⱼ*λᵢ₋₂₋ₖ, [i,j], [k], [(j,i-2)], [], [sc_Kₖⱼ,sc_λᵢ₋₂₋ₖ], sum)
R3 = RecurrentRelation((i-j)*(i-j-1)*Kᵢⱼ - (j+2)*(j+1)*Kᵢ₍ⱼ₊₂₎ ~ B₍ᵢ₋₂₎ⱼ, [i,j], 
                       [(2,:∞),(0,i-2)], [sc_Kᵢⱼ, sc_Kᵢ₍ⱼ₊₂₎], [ef_B₍ᵢ₋₂₎ⱼ]);

rs = RecurrentSeries{Float64}(:K, (1,), [x,y], [0,0], [R1, R2, R3])
compute_coefficients(rs, 3)

@test rs.coefficients[1] ≈ [0,0,0,0,-1/4,0,0,0,0,0]