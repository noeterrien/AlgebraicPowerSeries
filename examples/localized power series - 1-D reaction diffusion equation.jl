### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 958aa720-6b00-11f1-97e7-addf231645d5
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ 0ebca59a-1fe9-4b5b-8a0e-fb2f341fcfe3
using GLMakie

# ╔═╡ 32c37df6-a76b-4981-bbe5-5796d033af63
using Symbolics

# ╔═╡ 23c045c7-b0ab-44db-b2ea-5da6fdbeb3ae
using PlutoUI

# ╔═╡ e849f7a2-4eda-4647-937a-cc2b81064b48
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ 0b9f37e4-65b7-4fc2-9e2b-1749b58d744a
md"""
# Imports
"""

# ╔═╡ 3ce289cf-8066-41dc-bb64-ed55cf88a63d
import Latexify

# ╔═╡ 9e754812-488d-482e-a478-b6e1177f9c1e
md"""
# Variables
"""

# ╔═╡ dce75a36-ca33-4f49-92aa-9ca2cb553f80
@variables x y; nothing

# ╔═╡ 6c5a21e1-f856-45e7-acc7-026d1157f1f0
∂²x, ∂²y = Differential(x)^2, Differential(y)^2; nothing

# ╔═╡ 6f5522ad-ac7a-40d5-a192-5cc3c583f360
md"""
# Display initialization
"""

# ╔═╡ 7cb60079-651c-4786-85ca-e65bcf2b85b3
fig = Figure(); nothing

# ╔═╡ ef572240-fc81-4dd4-822e-cfc10fc5bbbe
begin
	ax1 = Axis(fig[1,1]; title="K(1,y) for different values of N with K centered at (0,0) - computed with PDESeries", xlabel="y")
	ax2 = Axis(fig[2,1]; title="K(1,y) for different values of N with K centered at (0,0) - computed with LocalizedPDESeries", xlabel="y")
	ax3 = Axis(fig[1,3]; title="K(1,y) for different values of N with K centered at (1,0)", xlabel="y")
	ax4 = Axis(fig[2,3]; title="K(1,y) for different values of N with K centered at (1,1)", xlabel="y")
end

# ╔═╡ cda1436c-70f8-4ef0-a230-fc82a95437f7
md"""
# Parameters
"""

# ╔═╡ cc968a10-6505-4084-8574-8517a56b4ff4
orders = [5, 10, 20, 50, 100]

# ╔═╡ 258f2608-2168-46c1-ac19-1504231e5d24
maxOrder = maximum(orders)

# ╔═╡ a368bd3b-29f7-40b2-9309-f27ede9198b4
y_range = 0:0.01:1

# ╔═╡ 89e1e482-434f-47ff-990d-824372ee454b
c = 3; nothing

# ╔═╡ 486a2cd7-1123-4108-9ec5-aa4865c4f4c7
ε = 1; nothing

# ╔═╡ 545a326d-d29e-4b4d-8800-de93ba13b607
λ_expr = 3 + x^2*sin(3*x)

# ╔═╡ 35c0258c-d170-4753-9955-a62db61ca104
ts_args = (:λ, [x], [λ_expr]); nothing

# ╔═╡ ac895624-d86e-489a-af91-188c4b8c3a41
md"""
# First case : centered at (0,0)
"""

# ╔═╡ f5e4d170-3c05-4c85-9aeb-d51b1ca1062b
md"""
## With PDESeries
"""

# ╔═╡ eee922ec-a9cc-47fe-84ab-06607cb4db0c
let
	# define λ PowerSeries
	λ_ps = TaylorExpansionSeries{Float64}(ts_args..., [0])
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
	compute_coefficients!(K_ps, maxOrder; verbose=1)

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

# ╔═╡ 48c6ab4f-9997-4b1e-b1e7-a9c5736c7071
fig[1,2] = Legend(fig, ax1, "orders", framevisible=false)

# ╔═╡ eea543f9-1568-440a-a7f7-875d84547c9c
md"""
## With LocalizedPDESeries
"""

# ╔═╡ 745d1e89-4ff9-44a1-b8d7-33af4efe27f0
let
	center = [0,0]
	
	# define λ PowerSeries
	λ_ps = TaylorExpansionSeries{Float64}(ts_args..., [0])
	compute_coefficients!(λ_ps, maxOrder)
	λ = SymbolicSeries(λ_ps)(x)

	# PDE and boundary conditions
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, center)

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ + c, 0, x)

	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ+c)/ε * K(x,y)

	# PDESeries initialization
	K_ps = LocalizedPDESeries{Float64}(:K, [x,y], center, [BC1, BC2, PDE], unknown)


	# compute coefficients and resulting polynomials
	Ks = []
	for order in orders
		compute_coefficients!(K_ps, order; verbose=1)

		local K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	# display result
	for (order, f) in zip(orders, Ks)
		lines!(ax2, y_range, f.(y_range); label="N = $order")
	end
	
end; nothing

# ╔═╡ a6cb3b35-0981-425c-be33-43850dc0ce05
fig[2,2] = Legend(fig, ax2, "orders", framevisible=false)

# ╔═╡ 7a750b6f-b202-4faa-ad50-559998d1c49b
md"""
# Second case : Centered at (1,0)
"""

# ╔═╡ 457f37e0-5978-4f7b-8b8f-5c1deb0e37d4
let
	center = [1,0]
	
	# define λ PowerSeries
	λ_ps_x = TaylorExpansionSeries{Float64}(ts_args..., [1])
	compute_coefficients!(λ_ps_x, maxOrder)
	λ_x = SymbolicSeries(λ_ps_x)(x)
	
	λ_ps_y = TaylorExpansionSeries{Float64}(ts_args..., [0])
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
		compute_coefficients!(K_ps, order; verbose=1)

		local K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	# display result
	for (order, f) in zip(orders, Ks)
		lines!(ax3, y_range, f.(y_range); label="N = $order")
	end
	
end; nothing

# ╔═╡ dc799b9b-5d3c-4d0f-8478-4711823cc067
fig[1,4] = Legend(fig, ax3, "orders", framevisible=false)

# ╔═╡ 74240b2e-8199-4821-a48b-0b9a6017eee7
md"""
# Third case : Centered at (1,1)
"""

# ╔═╡ ee211dc3-b028-4d85-83b9-2b0de4def53d
let
	center = [1,1]
	
	# define λ PowerSeries
	λ_ps = TaylorExpansionSeries{Float64}(ts_args..., [1])
	compute_coefficients!(λ_ps, maxOrder)
	λ = SymbolicSeries(λ_ps)

	# PDE and boundary conditions
	unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, center)

	BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, 0, x)

	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ(y)+c)/ε * K(x,y)

	# PDESeries initialization
	K_ps = LocalizedPDESeries{Float64}(:K, [x,y], center, [BC1, BC2, PDE], unknown)


	# compute coefficients and resulting polynomials
	Ks = []
	for order in orders
		compute_coefficients!(K_ps, order; verbose=1)

		local K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	# display result
	for (order, f) in zip(orders, Ks)
		lines!(ax4, y_range, f.(y_range); label="N = $order")
	end
	
end; nothing

# ╔═╡ edc76f07-1a85-4320-859a-806d1a8e5397
fig[2,4] = Legend(fig, ax4, "orders", framevisible=false)

# ╔═╡ 232aceb6-c2cb-4a1e-8d7d-43d6c61e5e2e
display(fig)

# ╔═╡ Cell order:
# ╟─0b9f37e4-65b7-4fc2-9e2b-1749b58d744a
# ╠═958aa720-6b00-11f1-97e7-addf231645d5
# ╠═0ebca59a-1fe9-4b5b-8a0e-fb2f341fcfe3
# ╠═32c37df6-a76b-4981-bbe5-5796d033af63
# ╠═23c045c7-b0ab-44db-b2ea-5da6fdbeb3ae
# ╠═e849f7a2-4eda-4647-937a-cc2b81064b48
# ╠═3ce289cf-8066-41dc-bb64-ed55cf88a63d
# ╟─9e754812-488d-482e-a478-b6e1177f9c1e
# ╠═dce75a36-ca33-4f49-92aa-9ca2cb553f80
# ╠═6c5a21e1-f856-45e7-acc7-026d1157f1f0
# ╟─6f5522ad-ac7a-40d5-a192-5cc3c583f360
# ╠═7cb60079-651c-4786-85ca-e65bcf2b85b3
# ╠═ef572240-fc81-4dd4-822e-cfc10fc5bbbe
# ╟─cda1436c-70f8-4ef0-a230-fc82a95437f7
# ╠═cc968a10-6505-4084-8574-8517a56b4ff4
# ╠═258f2608-2168-46c1-ac19-1504231e5d24
# ╠═a368bd3b-29f7-40b2-9309-f27ede9198b4
# ╠═89e1e482-434f-47ff-990d-824372ee454b
# ╠═486a2cd7-1123-4108-9ec5-aa4865c4f4c7
# ╠═545a326d-d29e-4b4d-8800-de93ba13b607
# ╠═35c0258c-d170-4753-9955-a62db61ca104
# ╟─ac895624-d86e-489a-af91-188c4b8c3a41
# ╟─f5e4d170-3c05-4c85-9aeb-d51b1ca1062b
# ╠═eee922ec-a9cc-47fe-84ab-06607cb4db0c
# ╠═48c6ab4f-9997-4b1e-b1e7-a9c5736c7071
# ╠═eea543f9-1568-440a-a7f7-875d84547c9c
# ╠═745d1e89-4ff9-44a1-b8d7-33af4efe27f0
# ╠═a6cb3b35-0981-425c-be33-43850dc0ce05
# ╟─7a750b6f-b202-4faa-ad50-559998d1c49b
# ╠═457f37e0-5978-4f7b-8b8f-5c1deb0e37d4
# ╠═dc799b9b-5d3c-4d0f-8478-4711823cc067
# ╟─74240b2e-8199-4821-a48b-0b9a6017eee7
# ╠═ee211dc3-b028-4d85-83b9-2b0de4def53d
# ╠═edc76f07-1a85-4320-859a-806d1a8e5397
# ╠═232aceb6-c2cb-4a1e-8d7d-43d6c61e5e2e
