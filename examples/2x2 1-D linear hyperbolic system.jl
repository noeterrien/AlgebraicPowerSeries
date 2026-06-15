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

# ╔═╡ c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ 9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
using GLMakie

# ╔═╡ 34c18021-d9bf-42d8-a9ba-001682b2cee5
using Symbolics

# ╔═╡ 9f9fd667-d45c-483b-af9a-393fe0ca3952
using PlutoUI

# ╔═╡ 5ad1546c-1481-42e2-8b99-db808eea7768
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
md"""
# Imports
"""

# ╔═╡ 4c1068dd-466a-4b9f-b37f-14c30e381571
import Latexify

# ╔═╡ 99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
md"""
# Variables
"""

# ╔═╡ 838b2173-6a55-4eb1-8a08-7cd98552d780
@variables x y

# ╔═╡ 23cfd29e-4cd9-49a0-ba93-338ed60b4e57
∂x, ∂y = Differential(x), Differential(y)

# ╔═╡ 902b946a-a877-42b3-8c18-81a4fc9a4262
md"""
# Parameters
"""

# ╔═╡ fa0e5bce-5ba3-4833-aed5-545e31d166de
@bind N PlutoUI.Slider(0:100; default=30, show_value=N -> "Order : N=$N")

# ╔═╡ c26326ea-ad12-4f6c-9dac-40bb68133c7f
q = 1; nothing

# ╔═╡ def0ec8e-e807-469d-b10f-425b56720a86
begin
	Σ_ps = TaylorExpansionSeries{Float64}(:Σ, [x], [1.2+x^3;0;;0;1.5+x^2], [0])
	compute_coefficients!(Σ_ps, N+1); 
	Σ = SymbolicSeries(Σ_ps)
end; nothing

# ╔═╡ 9cd56389-61d8-4641-828a-b46180e8e891
begin
	C_ps = TaylorExpansionSeries{Float64}(:C, [x], [3*cos(x);1+2*exp(x);;sin(2*x);1/(3+x^2)], [0])
	compute_coefficients!(C_ps, N)
	C = SymbolicSeries(C_ps)
	C₀ = SymbolicSeries{1}[C[1,1];0;;0;C[2,2]]
end; nothing

# ╔═╡ b4754ecb-8a1e-40b1-b30e-bca5592e7753
md"""
# PDE and boundary conditions
"""

# ╔═╡ 8a621f6e-8e1a-4789-8a2d-65a0a3693547
unknowns = selfseries_symbols(2,2); nothing

# ╔═╡ a301356b-432d-4528-85d0-56b207123a12
K = SymbolicSeries(unknowns, [0,0]); nothing

# ╔═╡ 79f8a834-6e59-4981-ad74-534429b26f99
md"""
### PDEs
"""

# ╔═╡ baef7a66-c86b-459e-a187-fecf1a43c059
PDEs = Σ(x)*∂x(K(x,y)) + ∂y(K(x,y))*Σ(y) .~ K(x,y)*(C(y)-∂y(Σ(y))) - C₀(x)*K(x,y)

# ╔═╡ 2e81893c-0de6-403f-8b15-72b6c46fadf6
md"""
### Boundary conditions
"""

# ╔═╡ b1569d5a-ba7e-4358-9f4f-f75417fdbeb8
BC1 = K(x,x)*Σ(x) - Σ(x)*K(x,x) .~ C(x) - C₀(x)

# ╔═╡ ac06dd23-4105-4a28-8517-badfc47b64f8
BC2 = K(x,0)*Σ(0)*[q, 1] .~ 0

# ╔═╡ 980fd8df-4003-400e-a96c-6199f855158e
md"""
# Computing the coefficients up to order N
"""

# ╔═╡ 3bbd8ae7-0f9d-41bb-b860-f2e0b982bafa
getindices(N) = N ≥ 1 ? generate_fullsym_indices(N-1, 2) : []; nothing

# ╔═╡ 44a14064-22fb-453f-996c-65aabac1b994
K_ps = PDESeries{Float64}(:K, [x,y], [0,0], unknowns, [BC1...; BC2...; PDEs...;], [N -> [], N -> [N], N -> [N], N -> [], N -> [N], N -> [N], getindices, getindices, getindices, getindices])

# ╔═╡ 505d68b5-1b8c-4e87-bc17-24deff01e5dc
compute_coefficients!(K_ps, N; verbose=2)

# ╔═╡ a44f22da-e181-4cd3-a04e-de3b050767ab
md"""
# Analyzing the results
"""

# ╔═╡ 993eaaab-6e9a-4af2-a5ed-3f19a77fb42d
orders = [15, 20, 25, 30]; nothing

# ╔═╡ a2269f12-b655-4a7f-9b8f-5ca20cbecb11
y_range = 0:0.01:1; nothing

# ╔═╡ a7d8d4f1-8b87-4ec8-8e81-4372f91e4162
begin 
	Ks = []
	for order in orders
		K_at_order = build_matrix_elt(K_ps, order)
		boundary_K = Matrix{Any}(undef, 2,2)
		for i in 1:2, j in 1:2
			Kˣˣ(y) = K_at_order[i,j](1,y)
			boundary_K[i,j] = Kˣˣ
		end
		push!(Ks, boundary_K)
	end
end

# ╔═╡ a8eb1972-97b6-486e-8f14-a49bcfecfed4
fig = Figure(); nothing

# ╔═╡ 81198cfe-3f86-438a-9bc3-95c4bf9b431d
labels = ["Kᵘᵘ(1,y)"; "Kᵛᵘ(1,y)";; "Kᵘᵛ(1,y)"; "Kᵛᵛ(1,y)"]; nothing

# ╔═╡ 23c47864-c772-45a7-a7ec-d7b3355bead9
axs = map(t -> Axis(fig[t[1],t[2]]; title="$(labels[t[1],t[2]]) for different values of N"), CartesianIndices((1:2, 1:2))); nothing

# ╔═╡ 692944ca-e7a4-4355-b054-572fef0b7bf7
for (order, K_array) in zip(orders, Ks)
	for i in 1:2, j in 1:2
		lines!(axs[i,j], y_range, K_array[i,j].(y_range); label="N=$order")
	end
end

# ╔═╡ 16a1886d-ced9-4d7a-a7cc-88bcdfdd3770
foreach(ax -> axislegend(ax; position=:rt), axs)

# ╔═╡ f37a2344-b532-4fa9-8002-90da20a98451
display(fig)

# ╔═╡ Cell order:
# ╟─b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
# ╠═c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
# ╠═5ad1546c-1481-42e2-8b99-db808eea7768
# ╠═4c1068dd-466a-4b9f-b37f-14c30e381571
# ╠═9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
# ╠═34c18021-d9bf-42d8-a9ba-001682b2cee5
# ╠═9f9fd667-d45c-483b-af9a-393fe0ca3952
# ╟─99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
# ╠═838b2173-6a55-4eb1-8a08-7cd98552d780
# ╠═23cfd29e-4cd9-49a0-ba93-338ed60b4e57
# ╟─902b946a-a877-42b3-8c18-81a4fc9a4262
# ╟─fa0e5bce-5ba3-4833-aed5-545e31d166de
# ╠═c26326ea-ad12-4f6c-9dac-40bb68133c7f
# ╠═def0ec8e-e807-469d-b10f-425b56720a86
# ╠═9cd56389-61d8-4641-828a-b46180e8e891
# ╟─b4754ecb-8a1e-40b1-b30e-bca5592e7753
# ╠═8a621f6e-8e1a-4789-8a2d-65a0a3693547
# ╠═a301356b-432d-4528-85d0-56b207123a12
# ╟─79f8a834-6e59-4981-ad74-534429b26f99
# ╠═baef7a66-c86b-459e-a187-fecf1a43c059
# ╟─2e81893c-0de6-403f-8b15-72b6c46fadf6
# ╠═b1569d5a-ba7e-4358-9f4f-f75417fdbeb8
# ╠═ac06dd23-4105-4a28-8517-badfc47b64f8
# ╟─980fd8df-4003-400e-a96c-6199f855158e
# ╠═3bbd8ae7-0f9d-41bb-b860-f2e0b982bafa
# ╠═44a14064-22fb-453f-996c-65aabac1b994
# ╠═505d68b5-1b8c-4e87-bc17-24deff01e5dc
# ╟─a44f22da-e181-4cd3-a04e-de3b050767ab
# ╠═993eaaab-6e9a-4af2-a5ed-3f19a77fb42d
# ╠═a2269f12-b655-4a7f-9b8f-5ca20cbecb11
# ╠═a7d8d4f1-8b87-4ec8-8e81-4372f91e4162
# ╠═a8eb1972-97b6-486e-8f14-a49bcfecfed4
# ╠═81198cfe-3f86-438a-9bc3-95c4bf9b431d
# ╠═23c47864-c772-45a7-a7ec-d7b3355bead9
# ╠═692944ca-e7a4-4355-b054-572fef0b7bf7
# ╠═16a1886d-ced9-4d7a-a7cc-88bcdfdd3770
# ╠═f37a2344-b532-4fa9-8002-90da20a98451
