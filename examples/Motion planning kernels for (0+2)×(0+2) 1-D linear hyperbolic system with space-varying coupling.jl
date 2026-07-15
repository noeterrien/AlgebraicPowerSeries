### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ def96e04-ec4e-4b63-aedf-a286d2ca2721
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ e213daa3-c0cb-41ca-9ed9-d0a7cc99058b
using GLMakie

# ╔═╡ 7184320f-9787-47ac-9877-974b440d2c5a
using Symbolics

# ╔═╡ 52cb1804-fb23-449e-b672-b206c6f5370a
using PlutoUI

# ╔═╡ c84d98d0-21cc-40db-b52d-1e4e31a52a37
using LinearAlgebra:Diagonal

# ╔═╡ d7a84177-544f-4214-ba90-737aa4b3a4ec
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ 796e8270-7f76-11f1-bf13-fb4cab8c5319
md"""
# Imports
"""

# ╔═╡ d190f97b-7a6a-40a3-8971-225fb90f2860
import Latexify

# ╔═╡ 5215f6be-4a5f-4244-a484-f395757c6b0a
md"""
# Variables
"""

# ╔═╡ f7175e13-b3e4-4441-9314-97fb9d631510
@variables x ξ; nothing

# ╔═╡ 66b09a75-cad3-4af9-9d4a-8e317d711bac
∂x, ∂ξ = Differential(x), Differential(ξ); nothing

# ╔═╡ 1d90a7de-7320-4e6a-a80e-953ed5adc521
md"""
# Parameters
"""

# ╔═╡ 2b7d4b33-2eec-4cb7-9cb6-9dc2baaa75aa
y_range = 0:0.001:1

# ╔═╡ 58ec8f2a-2c2c-4e74-a743-e5559f7a886c
N = 25

# ╔═╡ b81d654b-f991-4031-b620-2dda90600849
center = [0,0]

# ╔═╡ 97f62982-fbb1-4ff8-9a5d-259f8023b044
μ₁, μ₂ = 0.5, 0.3; nothing

# ╔═╡ 168445cc-4459-484a-8a6b-8cb38a058d34
begin
	σ_ts = TaylorExpansionSeries{Float64}(:σ, [x], [0; 0.3 + x^2/3 ;; 0.2 + x/3 ; 0], [0])
	compute_coefficients!(σ_ts, N)
	σ = SymbolicSeries(σ_ts)
	σ₁₂, σ₂₁ = σ[1,2], σ[2,1]
end

# ╔═╡ 78d74a82-2e16-4100-8e30-fa6223788da9
begin
	characteristic_ts = TaylorExpansionSeries{Float64}(:c, [x], [μ₂/μ₁*x], [0])
	compute_coefficients!(characteristic_ts, N)
	characteristic = SymbolicSeries(characteristic_ts)(x)
end

# ╔═╡ 5a225b94-0c31-46ee-be2f-6d2c4f0553dc
md"""
# PDE and boundary conditions
"""

# ╔═╡ 4f19ef93-032a-4119-b1a9-222c30659361
begin
	unknowns = selfseries_symbols(3, 2)
	L = SymbolicSeries(unknowns, center)
	L¹₁₁, L²₁₁, L₂₁, L¹₁₂, L²₁₂, L₂₂, = L
end; nothing

# ╔═╡ 95a23e87-c85e-4850-94f1-f4344b5087c3
PDEs = Diagonal([μ₁, μ₁, μ₂]) * ∂x(L(x, ξ)) + ∂ξ(L(x,ξ)) * Diagonal([μ₁, μ₂]) .~ L(x, ξ) * σ(ξ); nothing

# ╔═╡ 4e11dc31-5e6f-4bdf-94ff-ad7de8b724a8
BCs = [L¹₁₁(x, 0) ~ 0, 
       L¹₁₂(x, 0) ~ 0,
       L₂₂(x,0)   ~ 0,
       L²₁₂(x,x)  ~ σ₁₂(x)/(μ₂-μ₁),
       L₂₁(x,x)   ~ σ₂₁(x)/(μ₁-μ₂),
       L¹₁₁(x, characteristic) ~ L²₁₁(x, characteristic)]; nothing

# ╔═╡ cf2d4a94-c9e6-487d-a402-74577b4c33f4
md"""
# PDESeries
"""

# ╔═╡ d10b8732-fa79-43c0-b951-833bddca4d04
L_ps = PDESeries{Float64}(:L, [x, ξ], center, unknowns, [PDEs..., BCs...])

# ╔═╡ 9202062c-2da6-4436-915b-2423531f4586
compute_coefficients!(L_ps, N)

# ╔═╡ 4d23a20f-9dc1-4cca-98e6-079529d10175
md"""
# Resulting kernel
"""

# ╔═╡ 1b28015f-d337-47f9-9b81-cf641b4f1bc2
L_res = build_matrix_elt(L_ps); nothing

# ╔═╡ 2e2cd96b-84d5-4986-a121-241cceafbb27
L₁₁_res(ξ) = ξ < μ₂/μ₁ ? L_res[1,1](1, ξ) : L_res[2,1](1, ξ)

# ╔═╡ 2533bde3-0ae0-4e81-b55e-529f31ac0cc1
L₁₂_res(ξ) = ξ < μ₂/μ₁ ? L_res[1,2](1, ξ) : L_res[2,2](1, ξ)

# ╔═╡ 8850e81a-a20c-432c-a1b3-cb60717c8853
L₂₁_res(ξ) = L_res[3,1](1, ξ)

# ╔═╡ 6c353b73-7a47-48c5-8bb7-3cfce05c938a
L₂₂_res(ξ) = L_res[3,2](1, ξ)

# ╔═╡ 69989d5e-84f0-4e4f-82a5-b4eba8177099
L₂₁_other_res(x) = L_res[3,1](x, 0)

# ╔═╡ 81d2c8b1-ee49-408f-b9e2-071f0b60535a


# ╔═╡ d8acec43-d542-4026-93b9-dcfb14340c75
md"""
# Plot and display
"""

# ╔═╡ 598b18b4-5d99-4bbc-a4b3-a544fcd34e71
fig = Figure(); nothing

# ╔═╡ 31922153-bee1-434a-a518-7b61938a2ad8
ax = Axis(fig[1,1]; title="kernels for N=$N", xlabel="ξ")

# ╔═╡ 440a5da1-df93-4d30-951e-2a6c2690aa66
for (kernel, name) in zip([L₁₁_res, L₁₂_res, L₂₁_res, L₂₂_res, L₂₁_other_res], 
						  	["L₁₁(1, ξ)", "L₁₂(1, ξ)", "L₂₁(1, ξ)", "L₂₂(1, ξ)","L₂₁(ξ, 0)"])
	lines!(ax, y_range, kernel.(y_range); label=name)
end

# ╔═╡ 21d9881e-eae6-477b-a882-d74e93d143f2
fig[1,2] = Legend(fig, ax, "kernels", framevisible=false)

# ╔═╡ edd3f5df-68e6-4670-8e4b-2c9698acb825
display(fig)

# ╔═╡ Cell order:
# ╟─796e8270-7f76-11f1-bf13-fb4cab8c5319
# ╠═def96e04-ec4e-4b63-aedf-a286d2ca2721
# ╠═d7a84177-544f-4214-ba90-737aa4b3a4ec
# ╠═d190f97b-7a6a-40a3-8971-225fb90f2860
# ╠═e213daa3-c0cb-41ca-9ed9-d0a7cc99058b
# ╠═7184320f-9787-47ac-9877-974b440d2c5a
# ╠═52cb1804-fb23-449e-b672-b206c6f5370a
# ╠═c84d98d0-21cc-40db-b52d-1e4e31a52a37
# ╟─5215f6be-4a5f-4244-a484-f395757c6b0a
# ╠═f7175e13-b3e4-4441-9314-97fb9d631510
# ╠═66b09a75-cad3-4af9-9d4a-8e317d711bac
# ╟─1d90a7de-7320-4e6a-a80e-953ed5adc521
# ╠═2b7d4b33-2eec-4cb7-9cb6-9dc2baaa75aa
# ╠═58ec8f2a-2c2c-4e74-a743-e5559f7a886c
# ╠═b81d654b-f991-4031-b620-2dda90600849
# ╠═97f62982-fbb1-4ff8-9a5d-259f8023b044
# ╠═168445cc-4459-484a-8a6b-8cb38a058d34
# ╠═78d74a82-2e16-4100-8e30-fa6223788da9
# ╟─5a225b94-0c31-46ee-be2f-6d2c4f0553dc
# ╠═4f19ef93-032a-4119-b1a9-222c30659361
# ╠═95a23e87-c85e-4850-94f1-f4344b5087c3
# ╠═4e11dc31-5e6f-4bdf-94ff-ad7de8b724a8
# ╟─cf2d4a94-c9e6-487d-a402-74577b4c33f4
# ╠═d10b8732-fa79-43c0-b951-833bddca4d04
# ╠═9202062c-2da6-4436-915b-2423531f4586
# ╟─4d23a20f-9dc1-4cca-98e6-079529d10175
# ╠═1b28015f-d337-47f9-9b81-cf641b4f1bc2
# ╠═2e2cd96b-84d5-4986-a121-241cceafbb27
# ╠═2533bde3-0ae0-4e81-b55e-529f31ac0cc1
# ╠═8850e81a-a20c-432c-a1b3-cb60717c8853
# ╠═6c353b73-7a47-48c5-8bb7-3cfce05c938a
# ╠═69989d5e-84f0-4e4f-82a5-b4eba8177099
# ╠═81d2c8b1-ee49-408f-b9e2-071f0b60535a
# ╟─d8acec43-d542-4026-93b9-dcfb14340c75
# ╠═598b18b4-5d99-4bbc-a4b3-a544fcd34e71
# ╠═31922153-bee1-434a-a518-7b61938a2ad8
# ╠═440a5da1-df93-4d30-951e-2a6c2690aa66
# ╠═21d9881e-eae6-477b-a882-d74e93d143f2
# ╠═edd3f5df-68e6-4670-8e4b-2c9698acb825
