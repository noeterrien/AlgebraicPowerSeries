### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 96582742-0d69-47a1-bce6-5672385e4bb7
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ c481d149-fcb1-4e31-96ed-64abfd3704ea
using GLMakie

# ╔═╡ 61f3d5e2-45a2-40f1-bf23-b4dfacf7f0e3
using Symbolics

# ╔═╡ 527a13c9-61df-4f50-9179-e551218fc04d
using PlutoUI

# ╔═╡ ed023b55-5dd6-437c-8ff9-a83c471e1361
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ 776674bd-e446-4286-b6cf-643e0e9f1e5b
md"""
# Imports
"""

# ╔═╡ 243d9572-93af-4a74-bbe2-b260214cff31
import Latexify

# ╔═╡ 06d7d816-5bb2-4457-97cc-7d6b72524111
md"""
# Variables
"""

# ╔═╡ b69c5445-4d87-4986-bfcf-1309079bc36d
@variables x y; nothing

# ╔═╡ f3fa85ca-9d58-401f-83ec-4e2f68584436
∂²x, ∂²y = Differential(x)^2, Differential(y)^2; nothing

# ╔═╡ 3c8656ed-1fba-4d7d-bc05-bec290b2529a
md"""
# Parameters
"""

# ╔═╡ 388df369-4404-4295-ad17-d0b6bac16a24
y_range = 0:0.01:1

# ╔═╡ 1cdd8737-6f26-468c-b7ce-6a0802186b76
centers = [[0,0], [1,0], [1,1]]

# ╔═╡ 08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
c = 3; nothing

# ╔═╡ 7636cea2-c846-4c71-bba7-7168ef006283
ε = 1; nothing

# ╔═╡ d1504ce0-6b32-488b-9392-ba824b979c8e
λ_expr = √(0.5 + x^2)

# ╔═╡ 29ed4087-c9d0-47a8-9af6-cc7a70e1d039
orders1 = [4, 8, 15, 25, 50, 100]

# ╔═╡ d44ad123-66c1-4c97-b8e0-1c705b72073b
orders2 = [4, 8, 15, 25, 50]

# ╔═╡ 56c156b0-014e-43ff-b405-5d24d724dcad
maxOrder = maximum(orders1)

# ╔═╡ 839d6e0d-91e6-4bae-89bc-19ec7e192053
begin 
	λ0_ps = TaylorExpansionSeries{Float64}(:λ0, [x], [λ_expr], [0])
	compute_coefficients!(λ0_ps, maxOrder)
	λ0 = SymbolicSeries(λ0_ps)
end

# ╔═╡ 142db775-c6dd-41cf-aa24-f1c45be38752
begin 
	λ1_ps = TaylorExpansionSeries{Float64}(:λ1, [x], [λ_expr], [1])
	compute_coefficients!(λ1_ps, maxOrder)
	λ1 = SymbolicSeries(λ1_ps)
end

# ╔═╡ 9d45f916-8031-4208-b7ba-03c14020c293
md"""
# A useful function
"""

# ╔═╡ ac065b6a-389e-4b09-bfe8-1e7adf3c2764
function create_PDESeries(center, λ; loc=false)
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, center)

	println()
	println("creating PDESeries with center $center")

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, 0, x, x)
	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ K(x,y) * (λ(y)+c)/ε

	if loc
		LocalizedPDESeries{Float64}(:K, [x,y], center, [BC1, BC2, PDE], unknown)
	else
		PDESeries{Float64}(:K, [x,y], center, unknown, [BC1, BC2, PDE])
	end
end

# ╔═╡ 4305e556-a76c-4351-9c61-7f4c1d8daed7
md"""
# Display initialization
"""

# ╔═╡ 09ae54de-7a98-46a4-9f22-0efa1c3d6a0f
fig = Figure(); nothing

# ╔═╡ 014d214e-4a5e-4836-a068-dac223bcf6e5
titles = ["K at (0,0), λ at 0 (using PDESeries)",
		  "K at (0,0), λ at 0, then K translated to (1,0) with trunc_order=50",
		  "K at (0,0), λ at 0, then K translated to (1,0) with trunc_order=100",
		  "K at (1,0), λ at 0",
		  "K at (1,1), λ at 0",
		  "K at (0,0), λ at 0 (using LocalizedPDESeries)", 
		  "K at (1,0), λ at 1",
		  "K at (1,1), λ at 1",
		  "K at (0,0), λ at 1"]; nothing

# ╔═╡ 314bb6d4-d523-4eef-8ba2-d0ad86f3e9d7
axs = map(((idxf, idxt),) -> Axis(fig[idxf...]; title = titles[idxt], xlabel="y", limits=(nothing, nothing, -10, 10)), zip([(1,1), (2,1), (3,1), 
										  (1,3), (2,3), (3,3), 
										  (1,5), (2,5), (3,5)], eachindex(titles))); nothing

# ╔═╡ 1bee15ab-af94-4eb1-ae7a-a1a32adbf08c
md"""
# # PDESeries expanded at (0,0) with λ expanded around 0
# """

# ╔═╡ 8909a420-10e8-4f34-8adb-71713d06d3fa
begin 
	K1_ps = create_PDESeries(centers[1], λ0)
	compute_coefficients!(K1_ps, maxOrder)
end

# ╔═╡ 488cd860-2322-4f16-865b-cf43e4b4715d
# plot result
let
	Ks = []
	for order in orders1
		K, = build_matrix_elt(K1_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	for (order, f) in zip(orders1, Ks)
		lines!(axs[1], y_range, f.(y_range); label="N = $order")
	end
end

# ╔═╡ dd712221-2109-4f98-bd3d-c0a1353ab50b
# add legend
fig[1,2] = Legend(fig, axs[1], "orders", framevisible=false)

# ╔═╡ b748b0e1-b049-48d8-8ab3-453b74e4db6b
display(fig)

# ╔═╡ 54e051f4-a7e3-4177-a45a-8ba46a46cdf1
md"""
# Translated Series with K expanded around (0,0) and λ around 0, then K translated to (1,0)
"""

# ╔═╡ 2c51c206-4a9a-411e-9676-7ac66201eee5
md"""
## trunc_order = order
"""

# ╔═╡ f9094c92-2210-4ba7-9884-0888a3c147f6
K2_ps = TranslatedSeries(:K2, K1_ps, [1,0])

# ╔═╡ 9970cae3-8a24-43c4-9e9e-ae9364e1eeb5
# compute coefficients and resulting polynomials
begin
	Ks2 = []
	for order in orders2
		
		compute_coefficients!(K2_ps, order)
		K2, = build_matrix_elt(K2_ps, order)
		boundary_K2(y) = K2(1,y)				
		push!(Ks2, boundary_K2)

	end
end

# ╔═╡ b131bcd4-ae1b-4752-9f0e-7efee2b00590
# plot result
for (order, f) in zip(orders2, Ks2)
	lines!(axs[2], y_range, f.(y_range); label="N = $order")
end

# ╔═╡ eee0829b-3d39-4374-9378-44a6206bbeed
# legend
fig[2,2] = Legend(fig, axs[2], "orders", framevisible=false)

# ╔═╡ 7ffc5a02-b91a-426e-b4b6-eb82452c4f77
display(fig)

# ╔═╡ 9f958e1d-b0d8-4129-a01e-aa1ff7c1f175
md"""
## trunc_order=maxOrder1
"""

# ╔═╡ b89eb4f3-2e91-44fe-8fba-144d1e316cea
# compute coefficients and resulting polynomials
begin
	Ks3 = []
	for order in orders2
		
		compute_coefficients!(K2_ps, order; trunc_order=maxOrder)
		K3, = build_matrix_elt(K2_ps, order)
		boundary_K3(y) = K3(1,y)				
		push!(Ks3, boundary_K3)

	end
end

# ╔═╡ bd35c2bc-247c-40fe-8f91-1786e1cfc8dd
# plot result
for (order, f) in zip(orders2, Ks3)
	lines!(axs[3], y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 0727815c-d1c0-4170-a977-5669f5dd7c63
# legend
fig[3,2] = Legend(fig, axs[3], "orders", framevisible=false)

# ╔═╡ ce14727c-24fe-48cf-95c5-c0a5fce8dc4d
display(fig)

# ╔═╡ 18e561b9-4605-4ea0-983f-383c93cbf26f
md"""
# LocalizedPDESeries
"""

# ╔═╡ 91d31197-e6ae-45bf-9fc1-61db359ddb92
md"""
## Plotting function
"""

# ╔═╡ 6d9465e5-5ddd-446a-898e-ef892c25a135
function plot_locPDESeries(pdeseries, ax, leg_idx; show=false)
	
	# compute coefficients and resulting polynomials
	Ks4 = []
	for order in orders2
		
		compute_coefficients!(pdeseries, order; verbose=3)
		K4, = build_matrix_elt(pdeseries, order)
		boundary_K4(y) = K4(1,y)				
		push!(Ks4, boundary_K4)

	end
	
	# plot result
	for (order, f) in zip(orders2, Ks4)
		lines!(ax, y_range, f.(y_range); label="N = $order")
	end

	# legend
	fig[leg_idx...] = Legend(fig, ax, "orders", framevisible=false)

	if show
		display(fig)
	end

end

# ╔═╡ a7d7b324-9bfb-4e95-bc58-3a923712e3a5
md"""
## with λ expanded around 0
"""

# ╔═╡ 0dc56dfc-be90-4123-9452-f0f67cc5f819
for (c, ax, leg_idx) in zip(centers, axs[4:6], [(1,4),(2,4),(3,4)])
	pdeseries = create_PDESeries(c, λ0; loc=true)
	plot_locPDESeries(pdeseries, ax, leg_idx; show=true)
end

# ╔═╡ 92b4a667-22ac-46ac-970a-b0443eed6c6b
md"""
## with λ expanded around 1
"""

# ╔═╡ af006b90-6969-4636-9a88-442cfe8b4e08
for (c, ax, leg_idx) in zip(centers, axs[7:9], [(1,6),(2,6),(3,6)])
	pdeseries = create_PDESeries(c, λ1; loc=true)
	plot_locPDESeries(pdeseries, ax, leg_idx; show=true)
end

# ╔═╡ e424b71c-2535-44ef-a67d-36ebbd9864d8
md"""
# Save result
"""

# ╔═╡ fa5120e4-c62d-48fd-9005-1b4d78afae34
save("Fixing divergence in 1-D reaction diffusion with space varying reaction.png", fig; px_per_unit=4)

# ╔═╡ Cell order:
# ╟─776674bd-e446-4286-b6cf-643e0e9f1e5b
# ╠═96582742-0d69-47a1-bce6-5672385e4bb7
# ╠═ed023b55-5dd6-437c-8ff9-a83c471e1361
# ╠═243d9572-93af-4a74-bbe2-b260214cff31
# ╠═c481d149-fcb1-4e31-96ed-64abfd3704ea
# ╠═61f3d5e2-45a2-40f1-bf23-b4dfacf7f0e3
# ╠═527a13c9-61df-4f50-9179-e551218fc04d
# ╟─06d7d816-5bb2-4457-97cc-7d6b72524111
# ╠═b69c5445-4d87-4986-bfcf-1309079bc36d
# ╠═f3fa85ca-9d58-401f-83ec-4e2f68584436
# ╟─3c8656ed-1fba-4d7d-bc05-bec290b2529a
# ╠═388df369-4404-4295-ad17-d0b6bac16a24
# ╠═1cdd8737-6f26-468c-b7ce-6a0802186b76
# ╠═08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
# ╠═7636cea2-c846-4c71-bba7-7168ef006283
# ╠═d1504ce0-6b32-488b-9392-ba824b979c8e
# ╠═29ed4087-c9d0-47a8-9af6-cc7a70e1d039
# ╠═d44ad123-66c1-4c97-b8e0-1c705b72073b
# ╠═56c156b0-014e-43ff-b405-5d24d724dcad
# ╠═839d6e0d-91e6-4bae-89bc-19ec7e192053
# ╠═142db775-c6dd-41cf-aa24-f1c45be38752
# ╟─9d45f916-8031-4208-b7ba-03c14020c293
# ╠═ac065b6a-389e-4b09-bfe8-1e7adf3c2764
# ╟─4305e556-a76c-4351-9c61-7f4c1d8daed7
# ╠═09ae54de-7a98-46a4-9f22-0efa1c3d6a0f
# ╠═014d214e-4a5e-4836-a068-dac223bcf6e5
# ╠═314bb6d4-d523-4eef-8ba2-d0ad86f3e9d7
# ╟─1bee15ab-af94-4eb1-ae7a-a1a32adbf08c
# ╠═8909a420-10e8-4f34-8adb-71713d06d3fa
# ╠═488cd860-2322-4f16-865b-cf43e4b4715d
# ╠═dd712221-2109-4f98-bd3d-c0a1353ab50b
# ╠═b748b0e1-b049-48d8-8ab3-453b74e4db6b
# ╟─54e051f4-a7e3-4177-a45a-8ba46a46cdf1
# ╟─2c51c206-4a9a-411e-9676-7ac66201eee5
# ╠═f9094c92-2210-4ba7-9884-0888a3c147f6
# ╠═9970cae3-8a24-43c4-9e9e-ae9364e1eeb5
# ╠═b131bcd4-ae1b-4752-9f0e-7efee2b00590
# ╠═eee0829b-3d39-4374-9378-44a6206bbeed
# ╠═7ffc5a02-b91a-426e-b4b6-eb82452c4f77
# ╟─9f958e1d-b0d8-4129-a01e-aa1ff7c1f175
# ╠═b89eb4f3-2e91-44fe-8fba-144d1e316cea
# ╠═bd35c2bc-247c-40fe-8f91-1786e1cfc8dd
# ╠═0727815c-d1c0-4170-a977-5669f5dd7c63
# ╠═ce14727c-24fe-48cf-95c5-c0a5fce8dc4d
# ╟─18e561b9-4605-4ea0-983f-383c93cbf26f
# ╟─91d31197-e6ae-45bf-9fc1-61db359ddb92
# ╠═6d9465e5-5ddd-446a-898e-ef892c25a135
# ╟─a7d7b324-9bfb-4e95-bc58-3a923712e3a5
# ╠═0dc56dfc-be90-4123-9452-f0f67cc5f819
# ╟─92b4a667-22ac-46ac-970a-b0443eed6c6b
# ╠═af006b90-6969-4636-9a88-442cfe8b4e08
# ╟─e424b71c-2535-44ef-a67d-36ebbd9864d8
# ╠═fa5120e4-c62d-48fd-9005-1b4d78afae34
