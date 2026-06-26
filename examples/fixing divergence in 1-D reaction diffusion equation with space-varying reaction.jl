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

# ╔═╡ 4305e556-a76c-4351-9c61-7f4c1d8daed7
md"""
# Display initialization
"""

# ╔═╡ 09ae54de-7a98-46a4-9f22-0efa1c3d6a0f
fig = Figure(); nothing

# ╔═╡ cfb26243-8734-4c5b-9652-2c1fef722b45
begin
	ax1 = Axis(fig[1,1]; title="K(1,y) for different values of N with K centered at (0,0)", xlabel="y", limits=(nothing, nothing, -10, 10))
	ax2 = Axis(fig[2,1]; title="K(1,y) for different values of N with K centered at (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))
	ax3 = Axis(fig[3,1]; title="K(1,y) for different values of N with K centered at (0,0) then translated to (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))
end

# ╔═╡ 3c8656ed-1fba-4d7d-bc05-bec290b2529a
md"""
# Parameters
"""

# ╔═╡ c00635e2-6915-4dbf-8675-ba16645f8182
N = 25

# ╔═╡ 388df369-4404-4295-ad17-d0b6bac16a24
y_range = 0:0.01:1

# ╔═╡ 08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
c = 3; nothing

# ╔═╡ 7636cea2-c846-4c71-bba7-7168ef006283
ε = 1; nothing

# ╔═╡ d1504ce0-6b32-488b-9392-ba824b979c8e
λ_expr = √(0.5 + x^2)

# ╔═╡ a594af08-a3a6-47ab-8372-79aa88f95450
λ_args = (:λ, [x], [λ_expr]); nothing

# ╔═╡ d0157777-96f4-4ff3-bdfe-c765efd12645
md"""
# Kernel centered at (0,0) : diverges
"""

# ╔═╡ 8f57994c-ef08-44b7-8b6a-58ef94bcc46f
orders = [20, 30, 40, 50]

# ╔═╡ e422ecab-0897-4189-bc94-5d18d7a503b6
maxOrder = maximum(orders)

# ╔═╡ 2bc1348c-6f46-4897-9314-f79c37b0786f
let	
	# define λ PowerSeries
	λ_ps = TaylorExpansionSeries{Float64}(λ_args..., [0])
	compute_coefficients!(λ_ps, maxOrder)
	λ = SymbolicSeries(λ_ps)

	# PDE and boundary conditions
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, [0,0])

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, x)

	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ(y)+c)/ε * K(x,y)

	# compute coefficients
	K_ps = PDESeries{Float64}(:K, [x,y], [0,0], unknown, [BC1, BC2, PDE])
	compute_coefficients!(K_ps, maxOrder)

	# display results
	Ks = []
	for order in orders
		local K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	for (order, f) in zip(orders, Ks)
		lines!(ax1, y_range, f.(y_range); label="N = $order")
	end
end; nothing

# ╔═╡ 45c21971-edd4-4475-9d31-1fad480ec500
fig[1,2] = Legend(fig, ax1, "orders", framevisible=false)

# ╔═╡ fd6f4173-0001-40bf-bd98-78124bc7c69d
md"""
# Kernel centered at (1,0) : converges
"""

# ╔═╡ d00f938f-702b-4af4-a0e0-dce1f556e390
let
	center = [1,0]
	
	# define λ PowerSeries
	λ_ps_x = TaylorExpansionSeries{Float64}(λ_args..., [1])
	compute_coefficients!(λ_ps_x, maxOrder)
	λ_x = SymbolicSeries(λ_ps_x)(x)
	
	λ_ps_y = TaylorExpansionSeries{Float64}(λ_args..., [0])
	compute_coefficients!(λ_ps_y, maxOrder)
	λ_y = SymbolicSeries(λ_ps_y)(y)

	# PDE and boundary conditions
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, center)

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ_x + c, 0, x)

	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ_y+c)/ε * K(x,y)

	# PDESeries initialization
	K_ps = LocalizedPDESeries{Float64}(:K, [x,y], center, [BC1, BC2, PDE], unknown)


	# compute coefficients and resulting polynomials
	Ks = []
	for order in orders
		compute_coefficients!(K_ps, order)

		local K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	# display result
	for (order, f) in zip(orders, Ks)
		lines!(ax2, y_range, f.(y_range); label="N = $order")
	end
	
end; nothing

# ╔═╡ 16c857b8-2208-491c-931e-13873f24fbb7
fig[2,2] = Legend(fig, ax2, "orders", framevisible=false)

# ╔═╡ 239eef32-cedf-4990-8c82-8f7462b91e98
md"""
# Kernel centered at (0,0) then translated
"""

# ╔═╡ 5a164e67-ad16-42b8-9b6c-fdcbb0dc4b61
let	
	# define λ PowerSeries
	λ_ps = TaylorExpansionSeries{Float64}(λ_args..., [0])
	compute_coefficients!(λ_ps, maxOrder)
	λ = SymbolicSeries(λ_ps)

	# PDE and boundary conditions
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, [0,0])

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, x)

	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ(y)+c)/ε * K(x,y)

	# initialize translated series
	K_ps = PDESeries{Float64}(:K, [x,y], [0,0], unknown, [BC1, BC2, PDE])
	tr_K_ps = TranslatedSeries(:K_tr, K_ps, [1,0])

	# compute coefficients and resulting polynomials
	Ks = []
	for order in orders
		compute_coefficients!(tr_K_ps, order; trunc_order=maxOrder)
		
		local K, = build_matrix_elt(tr_K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	# display result
	for (order, f) in zip(orders, Ks)
		lines!(ax3, y_range, f.(y_range); label="N = $order")
	end
	
end; nothing

# ╔═╡ 2b3c0e41-9b3a-4197-ac16-b141985ad8f4
fig[3,2] = Legend(fig, ax3, "orders", framevisible=false)

# ╔═╡ 0dc1e94b-61fd-4739-9dee-38f7e24946c1
display(fig)

# ╔═╡ 7da23a18-7df6-4060-8070-923a752b7f84
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0).png", fig; px_per_unit=4)

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
# ╟─4305e556-a76c-4351-9c61-7f4c1d8daed7
# ╠═09ae54de-7a98-46a4-9f22-0efa1c3d6a0f
# ╠═cfb26243-8734-4c5b-9652-2c1fef722b45
# ╟─3c8656ed-1fba-4d7d-bc05-bec290b2529a
# ╠═c00635e2-6915-4dbf-8675-ba16645f8182
# ╠═388df369-4404-4295-ad17-d0b6bac16a24
# ╠═08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
# ╠═7636cea2-c846-4c71-bba7-7168ef006283
# ╠═d1504ce0-6b32-488b-9392-ba824b979c8e
# ╠═a594af08-a3a6-47ab-8372-79aa88f95450
# ╟─d0157777-96f4-4ff3-bdfe-c765efd12645
# ╠═8f57994c-ef08-44b7-8b6a-58ef94bcc46f
# ╠═e422ecab-0897-4189-bc94-5d18d7a503b6
# ╠═2bc1348c-6f46-4897-9314-f79c37b0786f
# ╠═45c21971-edd4-4475-9d31-1fad480ec500
# ╟─fd6f4173-0001-40bf-bd98-78124bc7c69d
# ╠═d00f938f-702b-4af4-a0e0-dce1f556e390
# ╠═16c857b8-2208-491c-931e-13873f24fbb7
# ╟─239eef32-cedf-4990-8c82-8f7462b91e98
# ╠═5a164e67-ad16-42b8-9b6c-fdcbb0dc4b61
# ╠═2b3c0e41-9b3a-4197-ac16-b141985ad8f4
# ╠═0dc1e94b-61fd-4739-9dee-38f7e24946c1
# ╠═7da23a18-7df6-4060-8070-923a752b7f84
