### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 08acecf5-7c81-4c72-8dae-57a4ce2e3428
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ b12e8568-940e-4f40-809e-fcf49b1a7e48
using GLMakie

# ╔═╡ dc2a7c5b-d450-44cc-841a-34fceff33ea9
using Symbolics

# ╔═╡ 9baedd01-60cc-4bdb-93e6-2bf3a3377062
using PlutoUI

# ╔═╡ 0f71c6c0-6644-11f1-8e94-757510873acd
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ 7dc01e76-7004-4148-936c-fa3eb8b853c8
md"""
# Imports
"""

# ╔═╡ af4f3f34-0a91-4c9e-a20f-1f0918315e26
import Latexify

# ╔═╡ 68100a42-e1f7-434c-a293-50eb19f7e74d
md"""
# Variables
"""

# ╔═╡ 4fedf422-9676-41df-a582-11c2db6a42d1
@variables x ξ; nothing

# ╔═╡ 8256693d-34d3-4600-9500-16975a0979cf
∂x, ∂ξ = Differential(x), Differential(ξ); nothing

# ╔═╡ d07d88f1-235c-4b1a-a244-492a2ace3683
md"""
# Parameters
"""

# ╔═╡ 605ac073-5b48-4882-bd9f-08a1acef3c85
@bind N PlutoUI.Slider(0:100; default=30, show_value=N -> "Order : N=$N")

# ╔═╡ 77d844c9-48e8-48c1-af7b-6ce3c1ee8e91
q = 1; nothing

# ╔═╡ a7850b12-9100-484d-93a1-2138c35c1a79
begin
	μ_ps = TaylorExpansionSeries{Float64}(:μ, [x], [1.5+x^2], [0])
	compute_coefficients!(μ_ps, N)
	μ = SymbolicSeries(μ_ps)
end; nothing

# ╔═╡ c02af617-2d13-4e44-8df2-762d62bd58f4
begin
	ϵ_ps = TaylorExpansionSeries{Float64}(:ϵ, [x], [1.2+x^3], [0])
	compute_coefficients!(ϵ_ps, N+1)
	ϵ = SymbolicSeries(ϵ_ps)
end; nothing

# ╔═╡ 3dd0b62e-b6cb-4504-8fda-3ed25ed20936
begin
	C_ps = TaylorExpansionSeries{Float64}(:C, [x], [3*cos(3*x);1+2*cos(2*x);;
                                                	sin(2*x)  ;1/(3+x^2)    ], [0])
	compute_coefficients!(C_ps, N)
	C = SymbolicSeries(C_ps)
end; nothing

# ╔═╡ ee072874-63d0-4e04-8ad4-27c960f4cc10
md"""
# PDE and boundary conditions
"""

# ╔═╡ c2acf115-6586-4b88-966d-a845a86bc7f8
unknowns = selfseries_symbols(2); nothing

# ╔═╡ add1fed5-ef14-404c-9e4f-d97300220c75
md"""
### Boundary conditions
"""

# ╔═╡ 45a7be5a-69bf-4bfd-a2d6-3112bc975d9a
K = SymbolicSeries(unknowns, [0,0]); nothing

# ╔═╡ 9d060a86-ecba-4299-8747-1a83a21d0340
BC1 = μ(0)*K[2](x,0) ~ q*ϵ(0)*K[1](x,0); nothing

# ╔═╡ c5ff454a-e338-4b81-ba94-37a8e0efef23
BC2 = (ϵ(x)+μ(x))*K[1](x,x) ~ -C[2,1](x); nothing

# ╔═╡ 56eb6e55-3513-4475-be1f-2e0cde89398f
md"""
### PDE
"""

# ╔═╡ 619d71ad-910c-42b7-aa72-79621e1c054b
PDE1 = μ(x)*∂x(K[2](x, ξ)) + μ(ξ)*∂ξ(K[2](x,ξ)) ~ -∂ξ(μ(ξ))*K[2](x, ξ) + 
													(C[1,2](ξ)-C[2,2](ξ))*K[1](x, ξ) +   C[2,2](x)*K[1](x, ξ); nothing

# ╔═╡ ede54c12-7516-434d-bba7-3d5d44ada895
PDE2 = μ(x)*∂x(K[1](x, ξ)) - ϵ(ξ)*∂ξ(K[1](x,ξ)) ~ ∂ξ(ϵ(ξ))*K[1](x, ξ) + 
													(C[2,1](ξ)-C[1,1](ξ))*K[2](x, ξ) +   C[2,2](x)*K[2](x, ξ); nothing

# ╔═╡ e5e9e7c5-25a4-4e18-9e4a-15f4e22a3e5a
md"""
# Computing the coefficients up to order N
"""

# ╔═╡ cf18d634-2635-477f-a3bb-61f0c850a34d
getindices(N) = N ≥ 1 ? generate_fullsym_indices(N-1, 2) : []; nothing

# ╔═╡ 64cc962f-503e-473f-babe-1c22339ca868
K_ps = PDESeries{Float64}(:K, [x,ξ], [0,0], unknowns, [BC1, BC2, PDE1, PDE2], [N -> [N], N -> [N], getindices, getindices])

# ╔═╡ 354c09a1-0883-4d85-a0d5-3dab6204525a
compute_coefficients!(K_ps, N; verbose=0)

# ╔═╡ e7889cc1-5bdb-46ae-ade5-559e652041d4
md"""
# Analyzing the results
"""

# ╔═╡ 36cf0c83-60f6-4c99-808c-8d0019af675a
orders = [15, 20, 25, 30]; nothing

# ╔═╡ 7ea46da6-a92f-463f-97b0-ed26e8aef2a2
ξ_range = 0:0.01:1; nothing

# ╔═╡ 829de2d6-49c1-4b64-8dc6-f8fe118fc12d
begin 
	Kᵛᵘs, Kᵛᵛs = [], []
	for order in orders
		Kᵛᵘ, Kᵛᵛ, = build_matrix_elt(K_ps, order)
		boundary_Kᵛᵘ(ξ) = Kᵛᵘ(1, ξ)
		boundary_Kᵛᵛ(ξ) = Kᵛᵛ(1, ξ)
		push!(Kᵛᵘs, boundary_Kᵛᵘ); push!(Kᵛᵛs, boundary_Kᵛᵛ)
	end
end

# ╔═╡ 9f3582ff-6464-44b4-88ab-29e9100deea7
fig = Figure(); nothing

# ╔═╡ 12033c22-7eb5-4f73-9b50-58ad577c7d1d
ax1 = Axis(fig[1,1]; title="Kᵛᵘ(1, ξ) for different values of N", xlabel="ξ", ylabel="Kᵛᵘ(1, ξ)"); nothing

# ╔═╡ 414bce49-a816-46de-927f-64519d562de8
ax2 = Axis(fig[1,2]; title="Kᵛᵛ(1, ξ) for different values of N", xlabel="ξ", ylabel="Kᵛᵛ(1, ξ)"); nothing

# ╔═╡ 2f7cc929-aaba-4450-8e09-f4e20faa72bd
for (order, Kᵛᵘ, Kᵛᵛ) in zip(orders, Kᵛᵘs, Kᵛᵛs)
	lines!(ax1, ξ_range, Kᵛᵘ.(ξ_range); label="N=$order")
	lines!(ax2, ξ_range, Kᵛᵛ.(ξ_range); label="N=$order")
end

# ╔═╡ eabf4807-8c68-41af-8822-e3d68d2b784e
axislegend(ax1, position=:lt); nothing

# ╔═╡ f4502167-fa88-48da-bb66-a6cb642d0a52
axislegend(ax2, position=:lt); nothing

# ╔═╡ 8c0e77e1-15d3-4ec9-a5f4-6fd34b630810
display(fig)

# ╔═╡ Cell order:
# ╟─7dc01e76-7004-4148-936c-fa3eb8b853c8
# ╠═08acecf5-7c81-4c72-8dae-57a4ce2e3428
# ╠═0f71c6c0-6644-11f1-8e94-757510873acd
# ╠═af4f3f34-0a91-4c9e-a20f-1f0918315e26
# ╠═b12e8568-940e-4f40-809e-fcf49b1a7e48
# ╠═dc2a7c5b-d450-44cc-841a-34fceff33ea9
# ╠═9baedd01-60cc-4bdb-93e6-2bf3a3377062
# ╟─68100a42-e1f7-434c-a293-50eb19f7e74d
# ╠═4fedf422-9676-41df-a582-11c2db6a42d1
# ╠═8256693d-34d3-4600-9500-16975a0979cf
# ╟─d07d88f1-235c-4b1a-a244-492a2ace3683
# ╟─605ac073-5b48-4882-bd9f-08a1acef3c85
# ╠═77d844c9-48e8-48c1-af7b-6ce3c1ee8e91
# ╠═a7850b12-9100-484d-93a1-2138c35c1a79
# ╠═c02af617-2d13-4e44-8df2-762d62bd58f4
# ╠═3dd0b62e-b6cb-4504-8fda-3ed25ed20936
# ╟─ee072874-63d0-4e04-8ad4-27c960f4cc10
# ╠═c2acf115-6586-4b88-966d-a845a86bc7f8
# ╟─add1fed5-ef14-404c-9e4f-d97300220c75
# ╠═45a7be5a-69bf-4bfd-a2d6-3112bc975d9a
# ╠═9d060a86-ecba-4299-8747-1a83a21d0340
# ╠═c5ff454a-e338-4b81-ba94-37a8e0efef23
# ╟─56eb6e55-3513-4475-be1f-2e0cde89398f
# ╠═619d71ad-910c-42b7-aa72-79621e1c054b
# ╠═ede54c12-7516-434d-bba7-3d5d44ada895
# ╟─e5e9e7c5-25a4-4e18-9e4a-15f4e22a3e5a
# ╠═cf18d634-2635-477f-a3bb-61f0c850a34d
# ╠═64cc962f-503e-473f-babe-1c22339ca868
# ╠═354c09a1-0883-4d85-a0d5-3dab6204525a
# ╟─e7889cc1-5bdb-46ae-ade5-559e652041d4
# ╠═36cf0c83-60f6-4c99-808c-8d0019af675a
# ╠═7ea46da6-a92f-463f-97b0-ed26e8aef2a2
# ╠═829de2d6-49c1-4b64-8dc6-f8fe118fc12d
# ╠═9f3582ff-6464-44b4-88ab-29e9100deea7
# ╠═12033c22-7eb5-4f73-9b50-58ad577c7d1d
# ╠═414bce49-a816-46de-927f-64519d562de8
# ╠═2f7cc929-aaba-4450-8e09-f4e20faa72bd
# ╠═eabf4807-8c68-41af-8822-e3d68d2b784e
# ╠═f4502167-fa88-48da-bb66-a6cb642d0a52
# ╠═8c0e77e1-15d3-4ec9-a5f4-6fd34b630810
