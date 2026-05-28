### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
begin
	import Pkg
	Pkg.add("Symbolics")
end; nothing

# ╔═╡ 5ad1546c-1481-42e2-8b99-db808eea7768
using Symbolics

# ╔═╡ 34c18021-d9bf-42d8-a9ba-001682b2cee5
using GLMakie

# ╔═╡ 4c1068dd-466a-4b9f-b37f-14c30e381571
include("../AlgebraicPowerSeries.jl"); nothing

# ╔═╡ b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
md"""
# Imports
"""

# ╔═╡ 9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
import Latexify

# ╔═╡ 99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
md"""
# Computing coefficients
"""

# ╔═╡ 089778d9-8560-49b8-8c3a-853df5536cfb
md"""
##### Defining variables and indices
"""

# ╔═╡ 838b2173-6a55-4eb1-8a08-7cd98552d780
@variables x y

# ╔═╡ 23cfd29e-4cd9-49a0-ba93-338ed60b4e57
@variables i j k

# ╔═╡ 902b946a-a877-42b3-8c18-81a4fc9a4262
md"""
##### Parameters
"""

# ╔═╡ fa0e5bce-5ba3-4833-aed5-545e31d166de
N = 30 # order

# ╔═╡ c26326ea-ad12-4f6c-9dac-40bb68133c7f
q = 1

# ╔═╡ be9ce343-7f72-48e6-8661-0a6a4c2c77f1
md"""
##### Σ function and coefficients
"""

# ╔═╡ def0ec8e-e807-469d-b10f-425b56720a86
begin
	ϵ₁ = TaylorExpansionSeries{Float64}(:ϵ₁, [x], [1.2+x^3], [0])
	ϵ₂ = TaylorExpansionSeries{Float64}(:ϵ₂, [x], [1.5+x^2], [0])
	compute_coefficients!(ϵ₁, N+1)
	compute_coefficients!(ϵ₂, N+1)
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
end; nothing

# ╔═╡ c47fa1af-af58-4b85-84ae-3d01c64a0b4a
md"""
##### C function and coefficients
"""

# ╔═╡ 9cd56389-61d8-4641-828a-b46180e8e891
begin
	c₁ = TaylorExpansionSeries{Float64}(:c₁, [x], [3*cos(x)], [0])
	c₂ = TaylorExpansionSeries{Float64}(:c₂, [x], [sin(2x)], [0])
	c₃ = TaylorExpansionSeries{Float64}(:c₃, [x], [1+2*exp(x)], [0])
	c₄ = TaylorExpansionSeries{Float64}(:c4, [x], [1/(3+x^2)], [0])
	compute_coefficients!(c₁, N)
	compute_coefficients!(c₂, N)
	compute_coefficients!(c₃, N)
	compute_coefficients!(c₄, N)
	@variables c¹ᵢ₋₁₋ⱼ₋ₖ c¹ⱼ₋ₖ c⁴ᵢ₋₁₋ⱼ₋ₖ c⁴ⱼ₋ₖ
	@variables c²ᵢ c²ⱼ₋ₖ c³ᵢ c³ⱼ₋ₖ
	c_c¹ᵢ₋₁₋ⱼ₋ₖ = SeriesCoefficient(c₁, c¹ᵢ₋₁₋ⱼ₋ₖ, [i-1-j-k], [i,j,k], (1,))
	c_c¹ⱼ₋ₖ     = SeriesCoefficient(c₁, c¹ⱼ₋ₖ    , [j-k]    , [j,k]  , (1,))
	c_c²ᵢ     	= SeriesCoefficient(c₂, c²ᵢ  , [i]  , [i]  , (1,))
	c_c²ⱼ₋ₖ   	= SeriesCoefficient(c₂, c²ⱼ₋ₖ, [j-k], [j,k], (1,))
	c_c³ᵢ     	= SeriesCoefficient(c₃, c³ᵢ  , [i]  , [i]  , (1,))
	c_c³ⱼ₋ₖ   	= SeriesCoefficient(c₃, c³ⱼ₋ₖ, [j-k], [j,k], (1,)) 
	c_c⁴ᵢ₋₁₋ⱼ₋ₖ = SeriesCoefficient(c₄, c⁴ᵢ₋₁₋ⱼ₋ₖ, [i-1-j-k], [i,j,k], (1,))
	c_c⁴ⱼ₋ₖ     = SeriesCoefficient(c₄, c⁴ⱼ₋ₖ    , [j-k]    , [j,k]  , (1,))
end; nothing

# ╔═╡ 0527f200-18c8-4098-b3ce-9dfaf7fb92ae
md"""
##### kernel coefficients
"""

# ╔═╡ 7cb2c546-9353-49a4-a14a-413d8c8dbdbe
begin
	@variables Kᵘᵘ₍ₖ₊ⱼ₎ⱼ Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ Kᵘᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵘᵘᵢ₀
	@variables Kᵘᵛ₍ₖ₊ⱼ₎ⱼ Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ Kᵘᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵘᵛᵢ₀ Kᵘᵛₖⱼ
	@variables Kᵛᵘ₍ₖ₊ⱼ₎ⱼ Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ Kᵛᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵛᵘᵢ₀ Kᵛᵘₖⱼ
	@variables Kᵛᵛ₍ₖ₊ⱼ₎ⱼ Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ Kᵛᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ Kᵛᵛᵢ₀

	c_Kᵘᵘ₍ₖ₊ⱼ₎ⱼ = SeriesCoefficient(:self, Kᵘᵘ₍ₖ₊ⱼ₎ⱼ, [k+j, j], [j,k], (1,1))
	c_Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ = SeriesCoefficient(:self, Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ, [k+j+1, j], [j,k], (1,1)) 
	c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ = SeriesCoefficient(:self, Kᵘᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, [i-1-j-k, k], [i,j,k], (1,1))
	c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ = SeriesCoefficient(:self, Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, [i-1-j+k, k], [i,j,k], (1,1))
	c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (1,1))
	c_Kᵘᵘᵢ₀ = SeriesCoefficient(:self, Kᵘᵘᵢ₀, [i, 0], [i], (1,1))

	c_Kᵘᵛ₍ₖ₊ⱼ₎ⱼ = SeriesCoefficient(:self, Kᵘᵛ₍ₖ₊ⱼ₎ⱼ, [k+j, j], [j,k], (2,1))
	c_Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ = SeriesCoefficient(:self, Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ, [k+j+1, j], [j,k], (1,2))
	c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ = SeriesCoefficient(:self, Kᵘᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, [i-1-j-k, k], [i,j,k], (1,2))
	c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ = SeriesCoefficient(:self, Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, [i-1-j+k, k], [i,j,k], (1,2))
	c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (1,2))
	c_Kᵘᵛᵢ₀ = SeriesCoefficient(:self, Kᵘᵛᵢ₀, [i, 0], [i], (1,2))
	c_Kᵘᵛₖⱼ = SeriesCoefficient(:self, Kᵘᵛₖⱼ, [k, j], [k,j], (1,2))

	c_Kᵛᵘ₍ₖ₊ⱼ₎ⱼ = SeriesCoefficient(:self, Kᵛᵘ₍ₖ₊ⱼ₎ⱼ, [k+j, j], [j,k], (1,2))
	c_Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ = SeriesCoefficient(:self, Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ, [k+j+1, j], [j,k], (2,1))
	c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ = SeriesCoefficient(:self, Kᵛᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, [i-1-j-k, k], [i,j,k], (2,1))
	c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ = SeriesCoefficient(:self, Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, [i-1-j+k, k], [i,j,k], (2,1))
	c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (2,1))
	c_Kᵛᵘᵢ₀ = SeriesCoefficient(:self, Kᵛᵘᵢ₀, [i, 0], [i], (2,1))
	c_Kᵛᵘₖⱼ = SeriesCoefficient(:self, Kᵛᵘₖⱼ, [k, j], [k,j], (2,1))

	c_Kᵛᵛ₍ₖ₊ⱼ₎ⱼ = SeriesCoefficient(:self, Kᵛᵛ₍ₖ₊ⱼ₎ⱼ, [k+j, j], [j,k], (2,2))
	c_Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ = SeriesCoefficient(:self, Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ, [k+j+1, j], [j,k], (2,2))
	c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ = SeriesCoefficient(:self, Kᵛᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, [i-1-j-k, k], [i,j,k], (2,2))
	c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ = SeriesCoefficient(:self, Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, [i-1-j+k, k], [i,j,k], (2,2))
	c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎ = SeriesCoefficient(:self, Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, [i-1-j+k+1, k+1] , [i,j,k], (2,2))
	c_Kᵛᵛᵢ₀ = SeriesCoefficient(:self, Kᵛᵛᵢ₀, [i, 0], [i], (2,2)) 
end; nothing

# ╔═╡ 73949d4a-9e28-4045-8977-c45fc17329ad
md"""
##### Relations of recurrence
"""

# ╔═╡ 2339fa53-55c3-47a5-8e67-4bcfe54ed54d
md"""
###### First relation
"""

# ╔═╡ 89fab97f-7a50-4319-aa4b-27af5a1d6ae5
begin
	R11 = RecurrentRelation(Kᵘᵘᵢ₀ ~ ϵ²₀/(q*ϵ¹₀)*Kᵘᵛᵢ₀, [i], [(0, :∞)], [c_Kᵘᵘᵢ₀, c_ϵ²₀, c_ϵ¹₀, c_Kᵘᵛᵢ₀], [])
	R12 = RecurrentRelation(Kᵛᵘᵢ₀ ~ ϵ²₀/(q*ϵ¹₀)*Kᵛᵛᵢ₀, [i], [(0, :∞)], [c_Kᵛᵘᵢ₀, c_ϵ²₀, c_ϵ¹₀, c_Kᵛᵛᵢ₀], [])
end; nothing

# ╔═╡ 7201eee7-96a2-4d9f-ae5d-3326017be86c
md"""
###### Second relation
"""

# ╔═╡ 0d8b4e73-b99c-4ecd-a38a-3005450aa69c
begin
	@variables ΣKᵘᵛₖⱼ ΣKᵛᵘₖⱼ ΣΣKᵘᵛₖⱼ ΣΣKᵛᵘₖⱼ
	EF211 = ExpandableFormula(:ΣKᵘᵛₖⱼ,  ΣKᵘᵛₖⱼ , Kᵘᵛₖⱼ, [k], [j], [(0,k)], [], [c_Kᵘᵛₖⱼ], sum)
	EF221 = ExpandableFormula(:ΣKᵛᵘₖⱼ,  ΣKᵛᵘₖⱼ , Kᵛᵘₖⱼ, [k], [j], [(0,k)], [], [c_Kᵛᵘₖⱼ], sum)
	EF21  = ExpandableFormula(:ΣΣKᵘᵛₖⱼ, ΣΣKᵘᵛₖⱼ, ΣKᵘᵛₖⱼ*(ϵ¹ᵢ₋ₖ + ϵ²ᵢ₋ₖ), [i], [k], [(0,i)], [EF211], [c_ϵ¹ᵢ₋ₖ, c_ϵ²ᵢ₋ₖ], sum)
	EF22  = ExpandableFormula(:ΣΣKᵛᵘₖⱼ, ΣΣKᵛᵘₖⱼ, ΣKᵛᵘₖⱼ*(ϵ¹ᵢ₋ₖ + ϵ²ᵢ₋ₖ), [i], [k], [(0,i)], [EF221], [c_ϵ¹ᵢ₋ₖ, c_ϵ²ᵢ₋ₖ], sum)
end; nothing

# ╔═╡ 5894eeff-f2eb-4f3a-8976-5fb74a145fc0
begin
	R21 = RecurrentRelation(ΣΣKᵘᵛₖⱼ ~ c²ᵢ, [i], [(0, :∞)], [c_c²ᵢ], [EF21])
	R22 = RecurrentRelation(ΣΣKᵛᵘₖⱼ ~ -c³ᵢ, [i], [(0, :∞)], [c_c³ᵢ], [EF22])
end; nothing

# ╔═╡ 2f923e1d-7945-4ce4-a785-d86fa3b272ed
md"""
###### Third relation
"""

# ╔═╡ 3f93969e-7570-4c00-aed3-6cb8e96517ef
begin
	@variables Σϵ¹xd_xKuu Σϵ¹yd_yKuu Σdϵ¹yKuu Σc³yKuv Σc¹xKuu Σc¹yKuu
	@variables Σϵ¹xd_xKuv Σϵ²yd_yKuv Σdϵ²yKuv Σc²yKuu Σc⁴xKuv Σc⁴yKuv
	@variables Σϵ²xd_xKvu Σϵ¹yd_yKvu Σdϵ¹yKvu Σc³yKvv Σc¹xKvu Σc¹yKvu
	@variables Σϵ²xd_xKvv Σϵ²yd_yKvv Σdϵ²yKvv Σc²yKvu Σc⁴xKvv Σc⁴yKvv
	
	EF311 = ExpandableFormula(:Σϵ¹xd_xKuu, Σϵ¹xd_xKuu, (k+1)*Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ¹ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵘ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ¹ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF312 = ExpandableFormula(:Σϵ¹yd_yKuu, Σϵ¹yd_yKuu, (k+1)*Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ¹ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ¹ⱼ₋ₖ], sum)
	EF313 = ExpandableFormula(:Σdϵ¹yKuu  , Σdϵ¹yKuu  , (j-k+1)*Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ¹ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ¹ⱼ₋ₖ₊₁], sum)
	EF314 = ExpandableFormula(:Σc³yKuv   , Σc³yKuv   , Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c³ⱼ₋ₖ            , [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c³ⱼ₋ₖ], sum)
	EF315 = ExpandableFormula(:Σc¹xKuu   , Σc¹xKuu   , Kᵘᵘ₍ₖ₊ⱼ₎ⱼ*c¹ᵢ₋₁₋ⱼ₋ₖ
	 , [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵘ₍ₖ₊ⱼ₎ⱼ, c_c¹ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF316 = ExpandableFormula(:Σc¹yKuu   , Σc¹yKuu   , Kᵘᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ*c¹ⱼ₋ₖ
	 , [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, c_c¹ⱼ₋ₖ], sum)
	
	EF321 = ExpandableFormula(:Σϵ¹xd_xKuv, Σϵ¹xd_xKuv, (k+1)*Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ¹ᵢ₋₁₋ⱼ₋ₖ, [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵛ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ¹ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF322 = ExpandableFormula(:Σϵ¹yd_yKuv, Σϵ²yd_yKuv, (k+1)*Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ²ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ²ⱼ₋ₖ], sum)
	EF323 = ExpandableFormula(:Σdϵ²yKuv  , Σdϵ²yKuv  , (j-k+1)*Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ²ⱼ₋ₖ₊₁, [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ²ⱼ₋ₖ₊₁], sum)
	EF324 = ExpandableFormula(:Σc²yKuu   , Σc²yKuu   , Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c²ⱼ₋ₖ, [i,j], [k], [(0,j)], [], [c_Kᵘᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c²ⱼ₋ₖ], sum)
	EF325 = ExpandableFormula(:Σc⁴xKuv   , Σc⁴xKuv   , Kᵘᵛ₍ₖ₊ⱼ₎ⱼ*c⁴ᵢ₋₁₋ⱼ₋ₖ
	 , [i,j], [k], [(0,i-1-j)], [], [c_Kᵘᵛ₍ₖ₊ⱼ₎ⱼ, c_c⁴ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF326 = ExpandableFormula(:Σc⁴yKuv   , Σc⁴yKuv   , Kᵘᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ*c⁴ⱼ₋ₖ
	 , [i,j], [k], [(0,j)], [], [c_Kᵘᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, c_c⁴ⱼ₋ₖ], sum)
	
	EF331 = ExpandableFormula(:Σϵ²xd_xKvu, Σϵ²xd_xKvu, (k+1)*Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ²ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵘ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ²ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF332 = ExpandableFormula(:Σϵ¹yd_yKvu, Σϵ¹yd_yKvu, (k+1)*Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ¹ⱼ₋ₖ, [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ¹ⱼ₋ₖ], sum)
	EF333 = ExpandableFormula(:Σdϵ¹yKvu  , Σdϵ¹yKvu  , (j-k+1)*Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ¹ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ¹ⱼ₋ₖ₊₁], sum)
	EF334 = ExpandableFormula(:Σc³yKvv   , Σc³yKvv   , Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c³ⱼ₋ₖ            , [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c³ⱼ₋ₖ], sum)
	EF335 = ExpandableFormula(:Σc¹xKvu   , Σc¹xKvu   , Kᵛᵘ₍ₖ₊ⱼ₎ⱼ*c¹ᵢ₋₁₋ⱼ₋ₖ
	 , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵘ₍ₖ₊ⱼ₎ⱼ, c_c¹ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF336 = ExpandableFormula(:Σc¹yKvu   , Σc¹yKvu   , Kᵛᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ*c¹ⱼ₋ₖ
	 , [i,j], [k], [(0,j)], [], [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, c_c¹ⱼ₋ₖ], sum)
	
	EF341 = ExpandableFormula(:Σϵ²xd_xKvv, Σϵ²xd_xKvv, (k+1)*Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ*ϵ²ᵢ₋₁₋ⱼ₋ₖ    , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵛ₍ₖ₊ⱼ₊₁₎ⱼ, c_ϵ²ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF342 = ExpandableFormula(:Σϵ²yd_yKvv, Σϵ²yd_yKvv, (k+1)*Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎*ϵ²ⱼ₋ₖ, [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₊₁₎₍ₖ₊₁₎, c_ϵ²ⱼ₋ₖ], sum)
	EF343 = ExpandableFormula(:Σdϵ²yKvv  , Σdϵ²yKvv  , (j-k+1)*Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*ϵ²ⱼ₋ₖ₊₁  , [i,j], [k], [(0,j)], [],     [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_ϵ²ⱼ₋ₖ₊₁], sum)
	EF344 = ExpandableFormula(:Σc²yKvu   , Σc²yKvu   , Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ*c²ⱼ₋ₖ            , [i,j], [k], [(0,j)], [],     [c_Kᵛᵘ₍ᵢ₋₁₋ⱼ₊ₖ₎ₖ, c_c²ⱼ₋ₖ], sum)
	EF345 = ExpandableFormula(:Σc⁴xKvv   , Σc⁴xKvv   , Kᵛᵛ₍ₖ₊ⱼ₎ⱼ*c⁴ᵢ₋₁₋ⱼ₋ₖ
	 , [i,j], [k], [(0,i-1-j)], [], [c_Kᵛᵛ₍ₖ₊ⱼ₎ⱼ, c_c⁴ᵢ₋₁₋ⱼ₋ₖ], sum)
	EF346 = ExpandableFormula(:Σc⁴yKvv   , Σc⁴yKvv   , Kᵛᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ*c⁴ⱼ₋ₖ
	 , [i,j], [k], [(0,j)], [], [c_Kᵛᵛ₍ᵢ₋₁₋ⱼ₋ₖ₎ₖ, c_c⁴ⱼ₋ₖ], sum)
end; nothing

# ╔═╡ 0bc50e88-7188-4ef6-a83c-4dc5e31c6092
begin
	R31 = RecurrentRelation(Σϵ¹xd_xKuu+Σϵ¹yd_yKuu ~ -Σdϵ¹yKuu - Σc³yKuv + Σc¹xKuu - Σc¹yKuu, [i,j], [(0, :∞), (0, i-1)], [], [EF311, EF312, EF313, EF314, EF315, EF316])
	R32 = RecurrentRelation(Σϵ¹xd_xKuv-Σϵ²yd_yKuv ~ Σdϵ²yKuv - Σc²yKuu + Σc⁴xKuv - Σc⁴yKuv, [i,j], [(0, :∞), (0, i-1)], [], [EF321, EF322, EF323, EF324, EF325, EF326])
	R33 = RecurrentRelation(Σϵ²xd_xKvu-Σϵ¹yd_yKvu ~ Σdϵ¹yKvu + Σc³yKvv + Σc¹yKvu - Σc¹xKvu, [i,j], [(0, :∞), (0, i-1)], [], [EF331, EF332, EF333, EF334, EF335, EF336])
	R34 = RecurrentRelation(Σϵ²xd_xKvv+Σϵ²yd_yKvv ~ -Σdϵ²yKvv + Σc²yKvu + Σc⁴yKvv - Σc⁴xKvv, [i,j], [(0, :∞), (0, i-1)], [], [EF341, EF342, EF343, EF344, EF345, EF346])
end; nothing

# ╔═╡ ebf696cb-3e04-4ead-b5fa-23c0991a8eb1
md"""
###### Series definition and computation
"""

# ╔═╡ 8009dbf2-4d50-4640-a22b-8c62cd841be3
begin
	rs = RecurrentSeries{Float64}(:crs, (2,2), [x,y], [0,0], [R11, R12, R21, R22, R31, R32, R33, R34])
	compute_coefficients!(rs, N; verbose=true)
end

# ╔═╡ c30e52dd-444e-43a3-bbb3-2b25b63311af
md"""
# Study of the series' convergence
"""

# ╔═╡ c191ce94-7a8a-4791-8449-cba849cc74ed
orders = [15, 20, 25, 30]

# ╔═╡ 4a36eaa3-93b9-4f58-a522-a17ea2566801
funcs = [ω -> build(rs, o)(1, ω) for o in orders]; nothing

# ╔═╡ d6e8bddc-851c-4195-994e-c8511ed27d82
begin 
	fig = Figure()
	axs = Matrix{Axis}(undef, 2,2)
	labels = ["Kᵘᵘ"; "Kᵛᵘ";; "Kᵘᵛ"; "Kᵛᵛ"]
	for i in axes(axs, 1), j in axes(axs, 2)
		axs[i,j] = Axis(fig[i,j]; xlabel="x", title="$(labels[i,j])(1,y)")
	end
end

# ╔═╡ f325fa04-1623-4a9d-9ad2-aed7557c7f4d
range = 0:0.05:1

# ╔═╡ 45b0e870-9d85-486e-841b-0dfc0da5655c
begin
	data = []
	for func in funcs
		push!(data, stack(func.(range), dims=3))
	end
end

# ╔═╡ 85fecc9a-a450-49c3-b426-d83d965d26ef
for (d,order) in zip(data,orders)
	for i in axes(axs, 1), j in axes(axs, 2)
		lines!(axs[i,j], range, d[i,j,:]; label="N=$order")
	end
end

# ╔═╡ f183a7c4-baaa-494f-9c9a-d944617b273d
foreach(ax -> axislegend(ax; position=:rt), axs)

# ╔═╡ 12b0551e-c67a-485b-80e0-6d00e91bf853
display(fig)

# ╔═╡ Cell order:
# ╟─b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
# ╠═c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
# ╠═5ad1546c-1481-42e2-8b99-db808eea7768
# ╠═4c1068dd-466a-4b9f-b37f-14c30e381571
# ╠═9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
# ╠═34c18021-d9bf-42d8-a9ba-001682b2cee5
# ╟─99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
# ╟─089778d9-8560-49b8-8c3a-853df5536cfb
# ╠═838b2173-6a55-4eb1-8a08-7cd98552d780
# ╠═23cfd29e-4cd9-49a0-ba93-338ed60b4e57
# ╟─902b946a-a877-42b3-8c18-81a4fc9a4262
# ╠═fa0e5bce-5ba3-4833-aed5-545e31d166de
# ╠═c26326ea-ad12-4f6c-9dac-40bb68133c7f
# ╟─be9ce343-7f72-48e6-8661-0a6a4c2c77f1
# ╠═def0ec8e-e807-469d-b10f-425b56720a86
# ╟─c47fa1af-af58-4b85-84ae-3d01c64a0b4a
# ╠═9cd56389-61d8-4641-828a-b46180e8e891
# ╟─0527f200-18c8-4098-b3ce-9dfaf7fb92ae
# ╠═7cb2c546-9353-49a4-a14a-413d8c8dbdbe
# ╟─73949d4a-9e28-4045-8977-c45fc17329ad
# ╟─2339fa53-55c3-47a5-8e67-4bcfe54ed54d
# ╠═89fab97f-7a50-4319-aa4b-27af5a1d6ae5
# ╟─7201eee7-96a2-4d9f-ae5d-3326017be86c
# ╠═0d8b4e73-b99c-4ecd-a38a-3005450aa69c
# ╠═5894eeff-f2eb-4f3a-8976-5fb74a145fc0
# ╟─2f923e1d-7945-4ce4-a785-d86fa3b272ed
# ╠═3f93969e-7570-4c00-aed3-6cb8e96517ef
# ╠═0bc50e88-7188-4ef6-a83c-4dc5e31c6092
# ╟─ebf696cb-3e04-4ead-b5fa-23c0991a8eb1
# ╠═8009dbf2-4d50-4640-a22b-8c62cd841be3
# ╟─c30e52dd-444e-43a3-bbb3-2b25b63311af
# ╠═c191ce94-7a8a-4791-8449-cba849cc74ed
# ╠═4a36eaa3-93b9-4f58-a522-a17ea2566801
# ╠═d6e8bddc-851c-4195-994e-c8511ed27d82
# ╠═f325fa04-1623-4a9d-9ad2-aed7557c7f4d
# ╠═45b0e870-9d85-486e-841b-0dfc0da5655c
# ╠═85fecc9a-a450-49c3-b426-d83d965d26ef
# ╠═f183a7c4-baaa-494f-9c9a-d944617b273d
# ╠═12b0551e-c67a-485b-80e0-6d00e91bf853
