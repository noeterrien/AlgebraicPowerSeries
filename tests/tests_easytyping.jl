using Test

include("../AlgebraicPowerSeries.jl")

@variables x y z
@variables i j k 

#parameters
N = 10
q = 1

# Σ function
Σ = TaylorExpansionSeries{Float64}(:Σ, [x], [1.2+x^3; 0 ;; 0 ; 1.5+x^2], [0])
compute_coefficients!(Σ, N+1); println("coefficients computed for Σ up to order $(N+1)")

# C function
C = TaylorExpansionSeries{Float64}(:C, [x], [3*cos(x);1+2*exp(x);;sin(2*x);1/(3+x^2)], [0])
compute_coefficients!(C, N); println("coefficients computed for C up to order $N")

# unknown series
K = selfseries_symbols(2,2)

# Relations of recurrence
R11 = RecurrentRelation(K[1,1][i,0].sym ~ Σ[2,2][0].sym/(q*Σ[1,1][0].sym)*K[1,2][i,0].sym, [i], [(0, :∞)], [K[1,1][i,0], Σ[2,2][0], Σ[1,1][0], K[1,2][i,0]], [])

@variables ΣKᵘᵛₖⱼ ΣΣKᵘᵛₖⱼ
EF211 = ExpandableFormula(:ΣKᵘᵛₖⱼ,  ΣKᵘᵛₖⱼ , K[1,2][k,j].sym, [k], [j], [(0,k)], [], [K[1,2][k,j]], sum)
EF21  = ExpandableFormula(:ΣΣKᵘᵛₖⱼ, ΣΣKᵘᵛₖⱼ, EF211.sym*(Σ[1,1][i-k].sym + Σ[2,2][i-k].sym), [i], [k], [(0,i)], [EF211], [Σ[1,1][i-k], Σ[2,2][i-k]], sum)
R21 = RecurrentRelation(EF21.sym ~ C[1,2][i].sym, [i], [(0, :∞)], [C[1,2][i]], [EF21])


ef = @∑ [i] (@∑ K[1,2][k,j] j in 0:k)*(Σ[1,1][i-k] + Σ[2,2][i-k]) k in 0:i#