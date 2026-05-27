using Test
using BenchmarkTools

include("../AlgebraicPowerSeries.jl")

@variables x y
@variables i j k

#----------------------------------1-D reaction diffusion equation with space-varying reaction------------------------------------

# ∑λᵢ = TaylorSeries{Float64}(:sin, [x], [sin(x)], [0])
# compute_coefficients!(∑λᵢ, 10)

# @variables Kᵢ₀ Kᵢⱼ Kᵢ₍ⱼ₊₂₎ Kₖⱼ
# @variables λᵢ₋₁ λᵢ₋₂₋ₖ
# @variables Σⱼ₌₀ⁱKᵢⱼ B₍ᵢ₋₂₎ⱼ

# sc_Kᵢ₀     = SeriesCoefficient(:self, Kᵢ₀, [i,0], [i], (1,))
# sc_Kᵢⱼ     = SeriesCoefficient(:self, Kᵢⱼ, [i,j], [i,j], (1,))
# sc_Kᵢ₍ⱼ₊₂₎ = SeriesCoefficient(:self, Kᵢ₍ⱼ₊₂₎, [i,j+2], [i,j], (1,))
# sc_Kₖⱼ     = SeriesCoefficient(:self, Kₖⱼ, [k,j], [k,j], (1,))

# sc_λᵢ₋₁ = SeriesCoefficient(∑λᵢ, λᵢ₋₁, [i-1], [i], (1,))
# sc_λᵢ₋₂₋ₖ = SeriesCoefficient(∑λᵢ, λᵢ₋₂₋ₖ, [i-2-k], [i,k], (1,))

# R1 = RecurrentRelation(Kᵢ₀ ~ 0, [i], [(0,:∞)], [sc_Kᵢ₀], [])

# sum(formulae::Vector) = +(formulae...)
# ef_Σⱼ₌₀ⁱKᵢⱼ = ExpandableFormula(:Σⱼ₌₀ⁱKᵢⱼ, Σⱼ₌₀ⁱKᵢⱼ, Kᵢⱼ, [i], [j], [(0,i)], [], [sc_Kᵢⱼ], sum)
# R2 = RecurrentRelation(Σⱼ₌₀ⁱKᵢⱼ ~ -λᵢ₋₁/(2*i), [i], [(1,:∞)], [sc_λᵢ₋₁], [ef_Σⱼ₌₀ⁱKᵢⱼ])



# ef_B₍ᵢ₋₂₎ⱼ = ExpandableFormula(:B₍ᵢ₋₂₎ⱼ , B₍ᵢ₋₂₎ⱼ, Kₖⱼ*λᵢ₋₂₋ₖ, [i,j], [k], [(j,i-2)], [], [sc_Kₖⱼ,sc_λᵢ₋₂₋ₖ], sum)
# R3 = RecurrentRelation((i-j)*(i-j-1)*Kᵢⱼ - (j+2)*(j+1)*Kᵢ₍ⱼ₊₂₎ ~ B₍ᵢ₋₂₎ⱼ, [i,j], 
#                        [(2,:∞),(0,i-2)], [sc_Kᵢⱼ, sc_Kᵢ₍ⱼ₊₂₎], [ef_B₍ᵢ₋₂₎ⱼ]);

# rs = RecurrentSeries{Float64}(:K, (1,), [x,y], [0,0], [R1, R2, R3])
# compute_coefficients!(rs, 3)

# @test rs.coefficients[1] ≈ [0,0,0,0,-1/4,0,0,0,0,0]

#-----------------------------------------------2x2 1-D linear hyperbolic system--------------------------------------------------
# parameters
N=50
q = 1

# Σ function and coefficients
ϵ₁ = TaylorSeries{Float64}(:ϵ₁, [x], [1+x^2], [0])
ϵ₂ = TaylorSeries{Float64}(:ϵ₂, [x], [exp(x)], [0])
compute_coefficients!(ϵ₁, N+1); println("coefficients computed for ϵ₁ up to order $(N+1)")
compute_coefficients!(ϵ₂, N+1); println("coefficients computed for ϵ₂ up to order $(N+1)")
@variables ϵ¹₀ ϵ¹ᵢ₋ₖ ϵ¹ᵢ₋₁₋ⱼ₋ₖ  ϵ¹ⱼ₋ₖ ϵ¹ⱼ₋ₖ₊₁
@variables ϵ²₀ ϵ²ᵢ₋ₖ ϵ²ᵢ₋₁₋ⱼ₋ₖ  ϵ²ⱼ₋ₖ ϵ²ⱼ₋ₖ₊₁
c_ϵ¹₀       = SeriesCoefficient(ϵ₁, ϵ¹₀      , [0]      , Num[]      , (1,))
c_ϵ¹ᵢ₋ₖ     = SeriesCoefficient(ϵ₁, ϵ¹ᵢ₋ₖ    , [i-k]    , [i,k]   , (1,))
c_ϵ¹ᵢ₋₁₋ⱼ₋ₖ = SeriesCoefficient(ϵ₁, ϵ¹ᵢ₋₁₋ⱼ₋ₖ, [i-1-j-k], [i,j,k] , (1,)) 
c_ϵ¹ⱼ₋ₖ     = SeriesCoefficient(ϵ₁, ϵ¹ⱼ₋ₖ    , [j-k]    , [j,k]   , (1,))
c_ϵ¹ⱼ₋ₖ₊₁   = SeriesCoefficient(ϵ₁, ϵ¹ⱼ₋ₖ₊₁  , [j-k+1]  , [j,k]   , (1,))
c_ϵ²₀       = SeriesCoefficient(ϵ₂, ϵ²₀      , [0]      , Num[]      , (1,))
c_ϵ²ᵢ₋ₖ     = SeriesCoefficient(ϵ₂, ϵ²ᵢ₋ₖ    , [i-k]    , [i,k]   , (1,))
c_ϵ²ᵢ₋₁₋ⱼ₋ₖ = SeriesCoefficient(ϵ₂, ϵ²ᵢ₋₁₋ⱼ₋ₖ, [i-1-j-k], [i,j,k] , (1,))
c_ϵ²ⱼ₋ₖ     = SeriesCoefficient(ϵ₂, ϵ²ⱼ₋ₖ    , [j-k]    , [j,k]   , (1,))
c_ϵ²ⱼ₋ₖ₊₁   = SeriesCoefficient(ϵ₂, ϵ²ⱼ₋ₖ₊₁  , [j-k+1]  , [j,k]   , (1,))

# C function and coefficients
c₂ = TaylorSeries{Float64}(:c₂, [x], [sin(x)], [0])
c₃ = TaylorSeries{Float64}(:c₃, [x], [cos(x)], [0])
compute_coefficients!(c₂, N); println("coefficients computed for c₂ up to order $N")
compute_coefficients!(c₃, N); println("coefficients computed for c₃ up to order $N")
@variables c²ᵢ c²ⱼ₋ₖ c³ᵢ c³ⱼ₋ₖ
c_c²ᵢ     = SeriesCoefficient(c₂, c²ᵢ  , [i]  , [i]  , (1,))
c_c²ⱼ₋ₖ   = SeriesCoefficient(c₂, c²ⱼ₋ₖ, [j-k], [j,k], (1,))
c_c³ᵢ     = SeriesCoefficient(c₃, c³ᵢ  , [i]  , [i]  , (1,))
c_c³ⱼ₋ₖ   = SeriesCoefficient(c₃, c³ⱼ₋ₖ, [j-k], [j,k], (1,))

# kernel coefficients
@variables Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵘᵘᵢ₀
@variables Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵘᵛᵢ₀ Kᵘᵛₖⱼ
@variables Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵛᵘᵢ₀ Kᵛᵘₖⱼ
@variables Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵛᵛᵢ₀

c_Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ         = SeriesCoefficient(:self, Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ        , [k+j+1, j]       , [j,k]  , (1,1)) 
c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ       = SeriesCoefficient(:self, Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ      , [i-1-j+k, k]     , [i,j,k], (1,1))
c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (1,1))
c_Kᵘᵘᵢ₀               = SeriesCoefficient(:self, Kᵘᵘᵢ₀              , [i, 0]           , [i]    , (1,1))
c_Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ         = SeriesCoefficient(:self, Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ        , [k+j+1, j]       , [j,k]  , (1,2))
c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ       = SeriesCoefficient(:self, Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ      , [i-1-j+k, k]     , [i,j,k], (1,2))
c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (1,2))
c_Kᵘᵛᵢ₀               = SeriesCoefficient(:self, Kᵘᵛᵢ₀              , [i, 0]           , [i]    , (1,2))
c_Kᵘᵛₖⱼ               = SeriesCoefficient(:self, Kᵘᵛₖⱼ              , [k, j]           , [k,j]  , (1,2))
c_Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ         = SeriesCoefficient(:self, Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ        , [k+j+1, j]       , [j,k]  , (2,1))
c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ       = SeriesCoefficient(:self, Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ      , [i-1-j+k, k]     , [i,j,k], (2,1))
c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (2,1))
c_Kᵛᵘᵢ₀               = SeriesCoefficient(:self, Kᵛᵘᵢ₀              , [i, 0]           , [i]    , (2,1))
c_Kᵛᵘₖⱼ               = SeriesCoefficient(:self, Kᵛᵘₖⱼ              , [k, j]           , [k,j]  , (2,1))
c_Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ         = SeriesCoefficient(:self, Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ        , [k+j+1, j]       , [j,k]  , (2,2))
c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ       = SeriesCoefficient(:self, Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ      , [i-1-j+k, k]     , [i,j,k], (2,2))
c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (2,2))
c_Kᵛᵛᵢ₀              = SeriesCoefficient(:self,  Kᵛᵛᵢ₀              , [i, 0]           , [i]    , (2,2)) 

# Relations of recurrence
R11 = RecurrentRelation(Kᵘᵘᵢ₀ ~ ϵ²₀/(q*ϵ¹₀)*Kᵘᵛᵢ₀, [i], [(0, :∞)], [c_Kᵘᵘᵢ₀, c_ϵ²₀, c_ϵ¹₀, c_Kᵘᵛᵢ₀], [])
R12 = RecurrentRelation(Kᵛᵘᵢ₀ ~ ϵ²₀/(q*ϵ¹₀)*Kᵛᵛᵢ₀, [i], [(0, :∞)], [c_Kᵛᵘᵢ₀, c_ϵ²₀, c_ϵ¹₀, c_Kᵛᵛᵢ₀], [])

@variables ΣKᵘᵛₖⱼ ΣKᵛᵘₖⱼ ΣΣKᵘᵛₖⱼ ΣΣKᵛᵘₖⱼ
EF211 = ExpandableFormula(:ΣKᵘᵛₖⱼ,  ΣKᵘᵛₖⱼ , Kᵘᵛₖⱼ, [k], [j], [(0,k)], [], [c_Kᵘᵛₖⱼ], sum)
EF221 = ExpandableFormula(:ΣKᵛᵘₖⱼ,  ΣKᵛᵘₖⱼ , Kᵛᵘₖⱼ, [k], [j], [(0,k)], [], [c_Kᵛᵘₖⱼ], sum)
EF21  = ExpandableFormula(:ΣΣKᵘᵛₖⱼ, ΣΣKᵘᵛₖⱼ, ΣKᵘᵛₖⱼ*(ϵ¹ᵢ₋ₖ + ϵ²ᵢ₋ₖ), [i], [k], [(0,i)], [EF211], [c_ϵ¹ᵢ₋ₖ, c_ϵ²ᵢ₋ₖ], sum)
EF22  = ExpandableFormula(:ΣΣKᵛᵘₖⱼ, ΣΣKᵛᵘₖⱼ, ΣKᵛᵘₖⱼ*(ϵ¹ᵢ₋ₖ + ϵ²ᵢ₋ₖ), [i], [k], [(0,i)], [EF221], [c_ϵ¹ᵢ₋ₖ, c_ϵ²ᵢ₋ₖ], sum)
R21 = RecurrentRelation(ΣΣKᵘᵛₖⱼ ~ c²ᵢ, [i], [(0, :∞)], [c_c²ᵢ], [EF21])
R22 = RecurrentRelation(ΣΣKᵛᵘₖⱼ ~ -c³ᵢ, [i], [(0, :∞)], [c_c³ᵢ], [EF22])

@variables Σϵ¹xd_xKuu Σϵ¹yd_yKuu Σdϵ¹yKuu Σc³yKuv 
@variables Σϵ¹xd_xKuv Σϵ²yd_yKuv Σdϵ²yKuv Σc²yKuu
@variables Σϵ²xd_xKvu Σϵ¹yd_yKvu Σdϵ¹yKvu Σc³yKvv
@variables Σϵ²xd_xKvv Σϵ²yd_yKvv Σdϵ²yKvv Σc²yKvu   
EF311 = ExpandableFormula(:Σϵ¹xd_xKuu, Σϵ¹xd_xKuu, (k+1)*Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ¹ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ¹ᵢ₋₁₋ⱼ₋ₖ], sum)
EF312 = ExpandableFormula(:Σϵ¹yd_yKuu, Σϵ¹yd_yKuu, (k+1)*Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ¹ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ¹ⱼ₋ₖ], sum)
EF313 = ExpandableFormula(:Σdϵ¹yKuu  , Σdϵ¹yKuu  , (j-k+1)*Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ¹ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ¹ⱼ₋ₖ₊₁], sum)
EF314 = ExpandableFormula(:Σc³yKuv   , Σc³yKuv   , Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c³ⱼ₋ₖ            , [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c³ⱼ₋ₖ], sum)
EF321 = ExpandableFormula(:Σϵ¹xd_xKuv, Σϵ¹xd_xKuv, (k+1)*Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ¹ᵢ₋₁₋ⱼ₋ₖ, [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ¹ᵢ₋₁₋ⱼ₋ₖ], sum)
EF322 = ExpandableFormula(:Σϵ¹yd_yKuv, Σϵ²yd_yKuv, (k+1)*Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ²ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ²ⱼ₋ₖ], sum)
EF323 = ExpandableFormula(:Σdϵ²yKuv  , Σdϵ²yKuv  , (j-k+1)*Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ²ⱼ₋ₖ₊₁, [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ²ⱼ₋ₖ₊₁], sum)
EF324 = ExpandableFormula(:Σc²yKuu   , Σc²yKuu   , Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c²ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c²ⱼ₋ₖ], sum)
EF331 = ExpandableFormula(:Σϵ²xd_xKvu, Σϵ²xd_xKvu, (k+1)*Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ²ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ²ᵢ₋₁₋ⱼ₋ₖ], sum)
EF332 = ExpandableFormula(:Σϵ¹yd_yKvu, Σϵ¹yd_yKvu, (k+1)*Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ¹ⱼ₋ₖ, [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ¹ⱼ₋ₖ], sum)
EF333 = ExpandableFormula(:Σdϵ¹yKvu  , Σdϵ¹yKvu  , (j-k+1)*Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ¹ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ¹ⱼ₋ₖ₊₁], sum)
EF334 = ExpandableFormula(:Σc³yKvv   , Σc³yKvv   , Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c³ⱼ₋ₖ            , [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c³ⱼ₋ₖ], sum)
EF341 = ExpandableFormula(:Σϵ²xd_xKvv, Σϵ²xd_xKvv, (k+1)*Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ²ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ²ᵢ₋₁₋ⱼ₋ₖ], sum)
EF342 = ExpandableFormula(:Σϵ²yd_yKvv, Σϵ²yd_yKvv, (k+1)*Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ²ⱼ₋ₖ, [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ²ⱼ₋ₖ], sum)
EF343 = ExpandableFormula(:Σdϵ²yKvv  , Σdϵ²yKvv  , (j-k+1)*Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ²ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ²ⱼ₋ₖ₊₁], sum)
EF344 = ExpandableFormula(:Σc²yKvu   , Σc²yKvu   , Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c²ⱼ₋ₖ            , [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c²ⱼ₋ₖ], sum)
R31 = RecurrentRelation(Σϵ¹xd_xKuu+Σϵ¹yd_yKuu ~ -Σdϵ¹yKuu - Σc³yKuv, [i,j], [(0, :∞), (0, i-1)], [], [EF311, EF312, EF313, EF314])
R32 = RecurrentRelation(Σϵ¹xd_xKuv-Σϵ²yd_yKuv ~ Σdϵ²yKuv - Σc²yKuu, [i,j], [(0, :∞), (0, i-1)], [], [EF321, EF322, EF323, EF324])
R33 = RecurrentRelation(Σϵ²xd_xKvu-Σϵ¹yd_yKvu ~ Σdϵ¹yKvu + Σc³yKvv, [i,j], [(0, :∞), (0, i-1)], [], [EF331, EF332, EF333, EF334])
R34 = RecurrentRelation(Σϵ²xd_xKvv+Σϵ²yd_yKvv ~ -Σdϵ²yKvv + Σc²yKvu, [i,j], [(0, :∞), (0, i-1)], [], [EF341, EF342, EF343, EF344])

coupled_rs = RecurrentSeries{Float64}(:crs, (2,2), [x,y], [0,0], [R11, R12, R21, R22, R31, R32, R33, R34])

# print("Coefficients computation for hyperbolic 2x2 series up to order $N takes on average : ")
# @btime compute_coefficients!(coupled_rs, N); 
compute_coefficients!(coupled_rs, N; verbose=true)
@test coupled_rs.coefficients[1,1][1:6] ≈ [0, 1/4, -1/4, 3/32, -7/16, 3/32]
@test coupled_rs.coefficients[1,2][1:6] ≈ [0, 1/4, 1/4, 3/32, -1/16, -9/32]
@test coupled_rs.coefficients[2,1][1:6] ≈ [-1/2, -1/8, 3/8, 5/64, 5/32, 17/64]
@test coupled_rs.coefficients[2,2][1:6] ≈ [-1/2, -1/8, 5/8, 5/64, 3/32, -43/64]

built_function = build(coupled_rs, 2)
@test built_function(0,0) ≈ [0;-1/2;;0;-1/2]
@test built_function(1,1) ≈ [-1/4;1/4;;1/4;-1/2]
@test built_function(1,2) ≈ [-21/32;101/64;;-13/32;-115/64]