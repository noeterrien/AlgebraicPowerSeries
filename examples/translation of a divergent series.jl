### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 29d7c8c0-8b2e-11f1-8edf-bd71dd0dfc84
begin
	import Pkg;
	Pkg.activate("@v1.12.6"); # change to your own version of your global julia environment or add the dependencies manually to the Pluto environment
end

# ╔═╡ 4161a285-d4b3-4cb7-84de-5b7789d505ff
using GLMakie

# ╔═╡ f23a6d47-f348-4782-9056-0ac77c2002c8
using Symbolics

# ╔═╡ 4b2b10a9-88b3-47e3-9442-4d7947dd6abf
using DoubleFloats

# ╔═╡ 5175e99d-4575-493f-af8f-749fee144c08
include("../AlgebraicPowerSeries.jl"); nothing;

# ╔═╡ 9049a2dc-55ea-4e6e-94e8-d07ce9674feb
import Latexify

# ╔═╡ 80bf6177-a3c8-494d-a739-2cbf860d2e57
md"""
# Variables and parameters
"""

# ╔═╡ a9adcac8-8e05-4bf0-87b6-9cd34ca05589
N = 40

# ╔═╡ fee93296-c301-46f4-bafc-40e580923f6b
orders = [2, 8, 15, 25]

# ╔═╡ 1daade11-ee9b-43d3-bad2-cc4dcd38793d
y_range = 0:0.01:1

# ╔═╡ 8924efb4-4b54-41b9-bc39-b1c6b1049f20
begin 
	variables = @variables x y
	∂²x, ∂²y = Differential(x)^2, Differential(y)^2
	center0 = [0,0]
	center1 = [1,1]
	center2 = [0.5, 0.5]
	ε = 1
	λ(x) = 3 + √(0.5+x^2)
end

# ╔═╡ 1d0cbe4a-f326-427c-b4eb-ad3e5456332e
fig = Figure(); nothing

# ╔═╡ fbe5d642-48b1-4b06-8bc4-7ea6a0b57e67
md"""
# K centered at (0,0)
"""

# ╔═╡ 2bc67f9a-234a-4251-8cd2-3b43228dc014
begin
	λ0_ps = TaylorExpansionSeries{Rational{BigInt}}(:λ0, [x], [λ(x)], [center0[2]])
	compute_coefficients!(λ0_ps, N)
	λ0 = SymbolicSeries(λ0_ps)
end

# ╔═╡ 19c1f2cb-dabf-493e-b502-b3b616779522
begin
	unknown0 = selfseries_symbols()
	K0 = SymbolicSeries(unknown0, center0)

	BC1_0 = K0(x,0) ~ 0
	BC2_0 = K0(x,x) ~ -1/(2*ε) * ∫(λ0(x), 0, x, x)
	PDE_0 = ∂²x(K0(x,y)) - ∂²y(K0(x,y)) ~ K0(x,y) * λ0(y)/ε

	K0_ps = PDESeries{Rational{BigInt}}(:K0, variables, center0, unknown0, [BC1_0, BC2_0, PDE_0])
end

# ╔═╡ 4c07cff6-9557-43dc-81d6-95c845d251d2
compute_coefficients!(K0_ps, N; solver=julia_default)

# ╔═╡ 5d2b5a88-5153-45a2-8a72-531727348337
md"""
## Plotting
"""

# ╔═╡ 03449854-0a2d-4951-85f1-b911690c3f7b
begin
	title0 = "K(1,y) with K expanded around (0,0) for different orders"
	ax0 = Axis(fig[1,1]; title=title0, xlabel="y", limits=(0,1,-2, 0))

	for order in orders
		K_built = build_matrix_elt(K0_ps, order)[1]
		K_bound(y) = K_built(1,y)
		lines!(ax0, y_range, K_bound.(y_range); label="N = $order")
	end
end

# ╔═╡ 35d70717-b974-4bbb-ab94-31fad0c3df15
md"""
# K centered at (1,1)
"""

# ╔═╡ d667d0a6-d0d2-4471-b562-970df4ad66fd
begin
	λ1_ps = TaylorExpansionSeries{Float64}(:λ1, [x], [λ(x)], [center1[2]])
	compute_coefficients!(λ1_ps, N)
	λ1 = SymbolicSeries(λ1_ps)
end

# ╔═╡ 2627f0a3-f8c2-4666-b0d0-26df0f3cfca0
begin
	unknown1 = selfseries_symbols()
	K1 = SymbolicSeries(unknown1, center1)

	BC1_1 = K1(x,0) ~ 0
	BC2_1 = K1(x,x) ~ -1/(2*ε) * ∫(λ1(x), 0, x, x)
	PDE_1 = ∂²x(K1(x,y)) - ∂²y(K1(x,y)) ~ K1(x,y) * λ1(y)/ε

	K1_ps = LocalizedPDESeries{Float64}(:K1, variables, center1, [BC1_1, BC2_1, PDE_1], unknown1)
end

# ╔═╡ 4149929f-c658-4b54-a6cc-bb0edf03f5d1
compute_coefficients!(K1_ps, N)

# ╔═╡ 92689787-59c1-4edd-beaf-f014e8ccb0b0
md"""
# Translation of K from (0,0) to (1,1)
"""

# ╔═╡ 4597bc59-525d-4188-8c37-bea04422930e
K1_trans_ps = TranslatedSeries(:K1_trans, K0_ps, center1)

# ╔═╡ 2ce3b258-4593-4bce-8c47-2ae9de7eba8e
md"""
## Plotting
"""

# ╔═╡ 584b81c1-3720-4bef-989a-db286f6b3a14
begin
	title1 = "K(1,y) with K translated from (0,0) to (1,1) for different orders"
	ax1 = Axis(fig[1,3]; title=title1, xlabel="y", limits=(0,1,-2, 0))

	for order in orders 
		compute_coefficients!(K1_trans_ps, order)
		local K_built = build_matrix_elt(K1_trans_ps, order)[1]
		local K_bound(y) = K_built(1,y)
		lines!(ax1, y_range, K_bound.(y_range); label="N=$order")
	end
end

# ╔═╡ 7dd5c6bd-bebe-4c8a-9785-ce30bfc14581
md"""
# Translation of K from (0,0) to (1,1) with a higher truncation order
"""

# ╔═╡ 03e09a88-c141-4d5a-b28f-94dd59138aed
begin
	title2 = "K(1,y) with K translated from (0,0) to (1,1) \n for different orders and truncation order is N=$N"
	ax2 = Axis(fig[2,1]; title=title2, xlabel="y", limits=(0,1,-3, 1))

	for order in orders 
		compute_coefficients!(K1_trans_ps, order; trunc_order=N)
		local K_built = build_matrix_elt(K1_trans_ps, order)[1]
		local K_bound(y) = K_built(1,y)
		lines!(ax2, y_range, K_bound.(y_range); label="N=$order")
	end
end

# ╔═╡ 6057c985-ac4d-47c2-957c-d873f5eb63f0
md"""
# Translation of K from (0,0) to (0.5, 0.5) with a higher truncation order
"""

# ╔═╡ a5c72752-1a53-4763-9b2f-6c35574aa276
K2_trans_ps = TranslatedSeries(:K2_trans, K0_ps, center2)

# ╔═╡ eebc6e11-0fad-42e7-a25e-0d90bfbe9e54
begin
	title3 = "K(1,y) with K translated from (0,0) to (0.5,0.5) \n for different orders and truncation order is N=$N"
	ax3 = Axis(fig[2,3]; title=title3, xlabel="y", limits=(0,1,-3, 1))

	for order in orders 
		compute_coefficients!(K2_trans_ps, order; trunc_order=N)
		local K_built = build_matrix_elt(K2_trans_ps, order)[1]
		local K_bound(y) = K_built(1,y)
		lines!(ax3, y_range, K_bound.(y_range); label="N=$order")
	end

	# expected results
	K_built = build_matrix_elt(K1_ps, N)[1]
	K_bound(y) = K_built(1,y)
	lines!(ax0, y_range, K_bound.(y_range); label="expected result (computed with localized \n power series at (1,1) and N = $N)", linestyle=:dash, color=:black)
	lines!(ax1, y_range, K_bound.(y_range); label="expected result (computed with localized \n power series at (1,1) and N = $N)", linestyle=:dash, color=:black)
	lines!(ax2, y_range, K_bound.(y_range); label="expected result (computed with localized \n power series at (1,1) and N = $N)", linestyle=:dash, color=:black)
	lines!(ax3, y_range, K_bound.(y_range); label="expected result (computed with localized \n power series at (1,1) and N = $N)", linestyle=:dash, color=:black)

	# legends
	Legend(fig[1,2], ax0, "orders", framevisible=false)
	Legend(fig[1,4], ax1, "orders", framevisible=false)
	Legend(fig[2,2], ax2, "orders", framevisible=false)
	Legend(fig[2,4], ax3, "orders", framevisible=false)
end

# ╔═╡ 0114c875-082e-494d-9ffe-eb68fcb47a32
md"""
# Display
"""

# ╔═╡ d5558441-8d2e-4e85-ac37-07c10733e123
display(fig)

# ╔═╡ Cell order:
# ╠═29d7c8c0-8b2e-11f1-8edf-bd71dd0dfc84
# ╠═5175e99d-4575-493f-af8f-749fee144c08
# ╠═9049a2dc-55ea-4e6e-94e8-d07ce9674feb
# ╠═4161a285-d4b3-4cb7-84de-5b7789d505ff
# ╠═f23a6d47-f348-4782-9056-0ac77c2002c8
# ╠═4b2b10a9-88b3-47e3-9442-4d7947dd6abf
# ╟─80bf6177-a3c8-494d-a739-2cbf860d2e57
# ╠═a9adcac8-8e05-4bf0-87b6-9cd34ca05589
# ╠═fee93296-c301-46f4-bafc-40e580923f6b
# ╠═1daade11-ee9b-43d3-bad2-cc4dcd38793d
# ╠═8924efb4-4b54-41b9-bc39-b1c6b1049f20
# ╠═1d0cbe4a-f326-427c-b4eb-ad3e5456332e
# ╟─fbe5d642-48b1-4b06-8bc4-7ea6a0b57e67
# ╠═2bc67f9a-234a-4251-8cd2-3b43228dc014
# ╠═19c1f2cb-dabf-493e-b502-b3b616779522
# ╠═4c07cff6-9557-43dc-81d6-95c845d251d2
# ╟─5d2b5a88-5153-45a2-8a72-531727348337
# ╠═03449854-0a2d-4951-85f1-b911690c3f7b
# ╟─35d70717-b974-4bbb-ab94-31fad0c3df15
# ╠═d667d0a6-d0d2-4471-b562-970df4ad66fd
# ╠═2627f0a3-f8c2-4666-b0d0-26df0f3cfca0
# ╠═4149929f-c658-4b54-a6cc-bb0edf03f5d1
# ╟─92689787-59c1-4edd-beaf-f014e8ccb0b0
# ╠═4597bc59-525d-4188-8c37-bea04422930e
# ╟─2ce3b258-4593-4bce-8c47-2ae9de7eba8e
# ╠═584b81c1-3720-4bef-989a-db286f6b3a14
# ╟─7dd5c6bd-bebe-4c8a-9785-ce30bfc14581
# ╠═03e09a88-c141-4d5a-b28f-94dd59138aed
# ╟─6057c985-ac4d-47c2-957c-d873f5eb63f0
# ╠═a5c72752-1a53-4763-9b2f-6c35574aa276
# ╠═eebc6e11-0fad-42e7-a25e-0d90bfbe9e54
# ╟─0114c875-082e-494d-9ffe-eb68fcb47a32
# ╠═d5558441-8d2e-4e85-ac37-07c10733e123
