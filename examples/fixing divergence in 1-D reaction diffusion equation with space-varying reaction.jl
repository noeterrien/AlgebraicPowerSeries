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

# ╔═╡ 3c8656ed-1fba-4d7d-bc05-bec290b2529a
md"""
# Parameters
"""

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

# ╔═╡ 8f57994c-ef08-44b7-8b6a-58ef94bcc46f
orders1 = [5, 10, 25, 50, 75, 100]

# ╔═╡ 2f2c3386-2f04-404b-adf4-8858f326c654
orders2 = [5, 10, 25, 50]

# ╔═╡ 346cfbd8-bce2-4cb6-8bfb-3d4951934683
orders3 = [40, 50, 60, 70]

# ╔═╡ d0157777-96f4-4ff3-bdfe-c765efd12645
md"""
# PDESeries centered at (0,0)
"""

# ╔═╡ fdb5820a-275b-45a2-81e7-102c630db571
md"""
## With PDESeries
"""

# ╔═╡ c00635e2-6915-4dbf-8675-ba16645f8182
maxOrder1 = maximum(orders1)

# ╔═╡ 7c7c8ed0-7885-43d7-90a7-ec5d16db77ba
begin
	λ_ps = TaylorExpansionSeries{Float64}(λ_args..., [0])
	compute_coefficients!(λ_ps, maxOrder1)
	λ = SymbolicSeries(λ_ps)
end

# ╔═╡ cbae4a9b-caf7-448c-b01e-db9b92eedf39
begin
	# PDE and boundary conditions
	unknown1 = selfseries_symbols()
	K1 = SymbolicSeries(unknown1, [0,0])

	BC1 = K1(x,0) ~ 0
	BC2 = K1(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, x)

	PDE = ∂²x(K1(x,y)) - ∂²y(K1(x,y)) ~ (λ(y)+c)/ε * K1(x,y)

	# compute coefficients
	K1_ps = PDESeries{Float64}(:K1, [x,y], [0,0], unknown1, [BC1, BC2, PDE])
	compute_coefficients!(K1_ps, maxOrder1)
end

# ╔═╡ 3293d0a7-6f98-428c-aee9-f3aa6167bb56
ax1 = Axis(fig[1,1]; title="K(1,y) with K centered at (0,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 6295029b-f446-4f42-a29f-45d867e62610
# plot results
let
	Ks = []
	for order in orders1
		K, = build_matrix_elt(K1_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	for (order, f) in zip(orders1, Ks)
		lines!(ax1, y_range, f.(y_range); label="N = $order")
	end
end

# ╔═╡ 45c21971-edd4-4475-9d31-1fad480ec500
fig[1,2] = Legend(fig, ax1, "orders", framevisible=false)

# ╔═╡ a2db92eb-ce1a-4be4-bbbd-c9faa6a5adb0
display(fig)

# ╔═╡ f0446ca1-9616-432a-acd6-9ea5cc33ff52
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0) - part0.png", fig; px_per_unit=4)

# ╔═╡ b85e5358-b174-4c88-905f-68967141d0b1
md"""
## With LocalizedPDESeries
"""

# ╔═╡ 32285d83-d1c2-4445-92b5-d1288e3c5b15
begin
	# PDE and boundary conditions
	unknown1bis = selfseries_symbols()
	K1bis = SymbolicSeries(unknown1bis, [0,0])

	BC1_bis = K1bis(x,0) ~ 0
	BC2_bis = K1bis(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, x)

	PDE_bis = ∂²x(K1bis(x,y)) - ∂²y(K1bis(x,y)) ~ (λ(y)+c)/ε * K1bis(x,y)

	# compute coefficients
	K1bis_ps = PDESeries{Float64}(:K1bis, [x,y], [0,0], unknown1bis, [BC1_bis, BC2_bis, PDE_bis])
	compute_coefficients!(K1bis_ps, maxOrder1)
end

# ╔═╡ fb5384b2-a83e-4ce7-b3ec-2f86db022670
ax1bis = Axis(fig[1,5]; title="K(1,y) with K centered at (0,0) - computed with LocalizedPDESeries", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ fcaa0859-e648-46eb-a277-4c9c68df80d4
# plot results
let
	Ks = []
	for order in orders1
		K, = build_matrix_elt(K1bis_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end

	for (order, f) in zip(orders1, Ks)
		lines!(ax1bis, y_range, f.(y_range); label="N = $order")
	end
end

# ╔═╡ a04aa4a8-f660-4e6d-8ab5-1c0c1ec27d5d
fig[1,6] = Legend(fig, ax1bis, "orders", framevisible=false)

# ╔═╡ e7065474-a09f-4743-a432-6a05004a5ad6
display(fig)

# ╔═╡ 030f33a3-d7bb-4fdc-a085-28ea1f890751
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0) - part0bis.png", fig; px_per_unit=4)

# ╔═╡ fd6f4173-0001-40bf-bd98-78124bc7c69d
md"""
# LocalizedPDESeries centered at (1,0)
"""

# ╔═╡ b0c9dc40-aba4-482b-bb39-f5e3a5f97771
center = [1,0]

# ╔═╡ 8871981d-24d5-427b-8e5d-d11b5aa75661
maxOrder2 = maximum(orders3)

# ╔═╡ 497e279a-79ae-4a97-88e7-8de9c89a6ced
begin
	λ_ps_x = TaylorExpansionSeries{Float64}(λ_args..., [1])
	compute_coefficients!(λ_ps_x, maxOrder2)
	λ_x = SymbolicSeries(λ_ps_x)(x)
end

# ╔═╡ 002a542b-0e09-49a2-853d-116666a16ecf
begin
	λ_ps_y = TaylorExpansionSeries{Float64}(λ_args..., [0])
	compute_coefficients!(λ_ps_y, maxOrder2)
	λ_y = SymbolicSeries(λ_ps_y)(y)
end

# ╔═╡ 45af7655-3dfb-4d35-8c94-76089b6a41e2
begin 
	# PDE and boundary conditions
	unknown2 = selfseries_symbols()
	K2 = SymbolicSeries(unknown2, center)

	BC1_2 = K2(x,0) ~ 0
	BC2_2 = K2(x,x) ~ -1/(2*ε) * ∫(λ_x + c, 0, x)

	PDE_2 = ∂²x(K2(x,y)) - ∂²y(K2(x,y)) ~ (λ_y+c)/ε * K2(x,y)

	# PDESeries initialization
	K2_ps = LocalizedPDESeries{Float64}(:K2, [x,y], center, [BC1_2, BC2_2, PDE_2], unknown2)
end

# ╔═╡ c7602bf2-ffbd-4b36-87d2-b124bed21d11
# compute coefficients and resulting polynomials
begin
	Ks2, Ks3 = Dict(), Dict()
	for order in orders2 ∪ orders3
		compute_coefficients!(K2_ps, order)

		let 
			K, = build_matrix_elt(K2_ps, order)
			boundary_K(y) = K(1,y)
			if order in orders2
				Ks2[order] = boundary_K
			end
			if order in orders3
				Ks3[order] = boundary_K
			end
		end
	end
end

# ╔═╡ 273dd6a3-140e-4ff7-b2c5-a54606e17360
ax2 = Axis(fig[2,1]; title="K(1,y) with K centered at (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 5752f576-1286-46c5-9888-ada84d660f24
# display result
for order in orders2
	f = Ks2[order]
	lines!(ax2, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 24566770-167e-4a3d-96f7-da0fa3276a27
fig[2,2] = Legend(fig, ax2, "orders", framevisible=false)

# ╔═╡ 0990246d-6302-43f9-9a40-9aa15bb3d209
display(fig)

# ╔═╡ 3df96d0b-c7a6-43d6-8724-541a19808fb4
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0)-part1.png", fig; px_per_unit=4)

# ╔═╡ b98b5c3e-7365-4575-8c2f-df39a49077de
ax3 = Axis(fig[3,1]; title="K(1,y) with K centered at (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ cfdb8055-001c-40b8-8b9a-570d90dde7c1
# display result
for order in orders3
	f = Ks3[order]
	lines!(ax3, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 7a8d9ce8-e369-4748-a477-750528a0d3c6
fig[3,2] = Legend(fig, ax3, "orders", framevisible=false)

# ╔═╡ 7e6d54b6-9ec7-4a27-807b-df51bf400469
display(fig)

# ╔═╡ 627bfa83-9545-4393-9ad5-6aadd4ee4df6
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0) - part2.png", fig; px_per_unit=4)

# ╔═╡ 239eef32-cedf-4990-8c82-8f7462b91e98
md"""
# PDESeries centered at (0,0) and translated to (1,0)
"""

# ╔═╡ 5f3389a8-6e59-455c-aa82-d24c8d4d1e5f
trunc_order1 = orders1[6]

# ╔═╡ 03e569b5-dfb3-4fcc-979d-b97e8218cb09
trunc_order2 = orders1[4]

# ╔═╡ 2f2ee767-ff71-4188-a388-8aacd60b35a9
tr_K_ps = TranslatedSeries(:K1_tr, K1_ps, [1,0])

# ╔═╡ fe19f8ab-4724-4ab3-9df5-5774e1068bc1
# compute coefficients and resulting polynomials
begin
	Ks4 = []
	for order in orders2
		
		compute_coefficients!(tr_K_ps, order)
		K4, = build_matrix_elt(tr_K_ps, order)
		boundary_K4(y) = K4(1,y)				
		push!(Ks4, boundary_K4)

	end
end

# ╔═╡ 3dded5a7-344c-49ed-a2db-65822d2e3063
ax4 = Axis(fig[1,3]; title="K(1,y) with K centered at (0,0) and translated to (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 8710aa04-7e07-4d7a-8685-5e57edf2139e
# display result
for (order, f) in zip(orders2, Ks4)
	lines!(ax4, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 4167bd16-e512-4b7b-b86b-e36a6dce5e4b
fig[1,4] = Legend(fig, ax4, "orders", framevisible=false)

# ╔═╡ a2ffc15b-6ecb-46ae-b035-94984b8e3deb
display(fig)

# ╔═╡ ccc78e3d-e3e8-4557-8909-bf532189e5d1
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0) - part3.png", fig; px_per_unit=4)

# ╔═╡ 6af46c86-e1ab-4289-8f7b-ec6b6eb40ae2
begin
	Ks5 = []
	for order in orders2
		compute_coefficients!(tr_K_ps, order; trunc_order=trunc_order1)
		K5, = build_matrix_elt(tr_K_ps, order)
		boundary_K5(y) = K5(1,y)				
		push!(Ks5, boundary_K5)
	end
end

# ╔═╡ 5c06cf81-34f0-4b91-aa80-cedc10a03e4e
ax5 = Axis(fig[2,3]; title="K(1,y) with K centered at (0,0) and translated to (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 4657c1aa-1628-419c-9e2e-862f49d53e93
# display result
for (order, f) in zip(orders2, Ks5)
	lines!(ax5, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ b94f0c3a-9166-4645-ac92-24261ab45eed
fig[2,4] = Legend(fig, ax5, "orders", framevisible=false)

# ╔═╡ b8e1de4d-d62b-47e4-a0b6-8f761e62705e
display(fig)

# ╔═╡ 6fde1005-b552-4c89-81bb-42957ebe4494
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0) - part4.png", fig; px_per_unit=4)

# ╔═╡ a5e57c56-2120-486e-a7ea-070d0eeb4d60
begin
	Ks6 = []
	for order in orders2
		compute_coefficients!(tr_K_ps, order; trunc_order=trunc_order2)
		K6, = build_matrix_elt(tr_K_ps, order)
		boundary_K6(y) = K6(1,y)				
		push!(Ks6, boundary_K6)
	end
end

# ╔═╡ 2d7594ae-7153-44d8-b609-9818ceb059ad
ax6 = Axis(fig[3,3]; title="K(1,y) with K centered at (0,0) and translated to (1,0)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 3faaeff1-aae5-4883-be84-6855ee105612
# display result
for (order, f) in zip(orders2, Ks6)
	lines!(ax6, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 50c8475e-0935-466b-ab70-5db99be4bac5
fig[3,4] = Legend(fig, ax6, "orders", framevisible=false)

# ╔═╡ 153c0d22-398d-40d9-a345-24ea3cd28798
display(fig)

# ╔═╡ b3d350f9-0273-4071-8d56-69765f550c62
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0)-part5.png", fig; px_per_unit=4)

# ╔═╡ 44895f8d-3876-4c92-a7cf-38ee02189a79
md"""
# LocalizedPDESeries centered at (1,1)
"""

# ╔═╡ 653e631e-c2d6-444b-abd3-b75638b2101a
center2 = [1,1]

# ╔═╡ 0f7590a0-11ee-4a2c-9236-efe494d15bc5
begin
	λ_ps_1 = TaylorExpansionSeries{Float64}(λ_args..., [1])
	compute_coefficients!(λ_ps_1, maxOrder1)
	λ_1 = SymbolicSeries(λ_ps_1)
end

# ╔═╡ bd1c3129-e1e3-45ea-8c4e-ee26e073fc09
begin 
	# PDE and boundary conditions
	unknown3 = selfseries_symbols()
	K3 = SymbolicSeries(unknown3, center2)

	BC1_3 = K3(x,0) ~ 0
	BC2_3 = K3(x,x) ~ -1/(2*ε) * ∫(λ_1(x) + c, 0, x)

	PDE_3 = ∂²x(K3(x,y)) - ∂²y(K3(x,y)) ~ (λ_1(y)+c)/ε * K3(x,y)

	# PDESeries initialization
	K3_ps = LocalizedPDESeries{Float64}(:K3, [x,y], center2, [BC1_3, BC2_3, PDE_3], unknown3)
end

# ╔═╡ 17c96edb-8a0b-436c-a6ed-d7dd8bc4c1c4
# compute coefficients and resulting polynomials
begin
	Ks7, Ks8 = Dict(), Dict()
	for order in orders2 ∪ orders3
		compute_coefficients!(K3_ps, order)

		let 
			K, = build_matrix_elt(K3_ps, order)
			boundary_K(y) = K(1,y)
			if order in orders2
				Ks7[order] = boundary_K
			end
			if order in orders3
				Ks8[order] = boundary_K
			end
		end
	end
end

# ╔═╡ 0dbe68f0-420c-48ac-bd5e-8720edd1dfbf
ax7 = Axis(fig[2,5]; title="K(1,y) with K centered at (1,1)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 9310dda2-a8b1-4cb8-9663-07cbce8f80c2
# display result
for order in orders2
	f = Ks7[order]
	lines!(ax7, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 4ad7cd3f-29d9-4bab-8cd4-1d3315a29e58
fig[2,6] = Legend(fig, ax7, "orders", framevisible=false)

# ╔═╡ a85b659d-e95a-4194-ac0c-fb49267b99de
display(fig)

# ╔═╡ e3daab95-21a5-4403-87dc-6faa67ff93a2
save("comparison between centered at (0,0), centered at (1,0) and translated from (0,0) to (1,0)-part6.png", fig; px_per_unit=4)

# ╔═╡ 9530a0b1-0922-409f-8af8-b91d186d0d2d
ax8 = Axis(fig[3,5]; title="K(1,y) with K centered at (1,1)", xlabel="y", limits=(nothing, nothing, -10, 10))

# ╔═╡ 57eced27-b9e4-4e12-a109-92c0897b5dee
# display result
for order in orders3
	f = Ks8[order]
	lines!(ax8, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ b223c591-ebda-4adb-86ff-b58fbfce050c
fig[3,6] = Legend(fig, ax8, "orders", framevisible=false)

# ╔═╡ e95f6771-984d-4f1d-ac24-f4554a84e431
md"""
# Display and save figure
"""

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
# ╟─3c8656ed-1fba-4d7d-bc05-bec290b2529a
# ╠═388df369-4404-4295-ad17-d0b6bac16a24
# ╠═08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
# ╠═7636cea2-c846-4c71-bba7-7168ef006283
# ╠═d1504ce0-6b32-488b-9392-ba824b979c8e
# ╠═a594af08-a3a6-47ab-8372-79aa88f95450
# ╠═8f57994c-ef08-44b7-8b6a-58ef94bcc46f
# ╠═2f2c3386-2f04-404b-adf4-8858f326c654
# ╠═346cfbd8-bce2-4cb6-8bfb-3d4951934683
# ╟─d0157777-96f4-4ff3-bdfe-c765efd12645
# ╟─fdb5820a-275b-45a2-81e7-102c630db571
# ╠═c00635e2-6915-4dbf-8675-ba16645f8182
# ╠═7c7c8ed0-7885-43d7-90a7-ec5d16db77ba
# ╠═cbae4a9b-caf7-448c-b01e-db9b92eedf39
# ╠═3293d0a7-6f98-428c-aee9-f3aa6167bb56
# ╠═6295029b-f446-4f42-a29f-45d867e62610
# ╠═45c21971-edd4-4475-9d31-1fad480ec500
# ╠═a2db92eb-ce1a-4be4-bbbd-c9faa6a5adb0
# ╠═f0446ca1-9616-432a-acd6-9ea5cc33ff52
# ╟─b85e5358-b174-4c88-905f-68967141d0b1
# ╠═32285d83-d1c2-4445-92b5-d1288e3c5b15
# ╠═fb5384b2-a83e-4ce7-b3ec-2f86db022670
# ╠═fcaa0859-e648-46eb-a277-4c9c68df80d4
# ╠═a04aa4a8-f660-4e6d-8ab5-1c0c1ec27d5d
# ╠═e7065474-a09f-4743-a432-6a05004a5ad6
# ╠═030f33a3-d7bb-4fdc-a085-28ea1f890751
# ╟─fd6f4173-0001-40bf-bd98-78124bc7c69d
# ╠═b0c9dc40-aba4-482b-bb39-f5e3a5f97771
# ╠═8871981d-24d5-427b-8e5d-d11b5aa75661
# ╠═497e279a-79ae-4a97-88e7-8de9c89a6ced
# ╠═002a542b-0e09-49a2-853d-116666a16ecf
# ╠═45af7655-3dfb-4d35-8c94-76089b6a41e2
# ╠═c7602bf2-ffbd-4b36-87d2-b124bed21d11
# ╠═273dd6a3-140e-4ff7-b2c5-a54606e17360
# ╠═5752f576-1286-46c5-9888-ada84d660f24
# ╠═24566770-167e-4a3d-96f7-da0fa3276a27
# ╠═0990246d-6302-43f9-9a40-9aa15bb3d209
# ╠═3df96d0b-c7a6-43d6-8724-541a19808fb4
# ╠═b98b5c3e-7365-4575-8c2f-df39a49077de
# ╠═cfdb8055-001c-40b8-8b9a-570d90dde7c1
# ╠═7a8d9ce8-e369-4748-a477-750528a0d3c6
# ╠═7e6d54b6-9ec7-4a27-807b-df51bf400469
# ╠═627bfa83-9545-4393-9ad5-6aadd4ee4df6
# ╟─239eef32-cedf-4990-8c82-8f7462b91e98
# ╠═5f3389a8-6e59-455c-aa82-d24c8d4d1e5f
# ╠═03e569b5-dfb3-4fcc-979d-b97e8218cb09
# ╠═2f2ee767-ff71-4188-a388-8aacd60b35a9
# ╠═fe19f8ab-4724-4ab3-9df5-5774e1068bc1
# ╠═3dded5a7-344c-49ed-a2db-65822d2e3063
# ╠═8710aa04-7e07-4d7a-8685-5e57edf2139e
# ╠═4167bd16-e512-4b7b-b86b-e36a6dce5e4b
# ╠═a2ffc15b-6ecb-46ae-b035-94984b8e3deb
# ╠═ccc78e3d-e3e8-4557-8909-bf532189e5d1
# ╠═6af46c86-e1ab-4289-8f7b-ec6b6eb40ae2
# ╠═5c06cf81-34f0-4b91-aa80-cedc10a03e4e
# ╠═4657c1aa-1628-419c-9e2e-862f49d53e93
# ╠═b94f0c3a-9166-4645-ac92-24261ab45eed
# ╠═b8e1de4d-d62b-47e4-a0b6-8f761e62705e
# ╠═6fde1005-b552-4c89-81bb-42957ebe4494
# ╠═a5e57c56-2120-486e-a7ea-070d0eeb4d60
# ╠═2d7594ae-7153-44d8-b609-9818ceb059ad
# ╠═3faaeff1-aae5-4883-be84-6855ee105612
# ╠═50c8475e-0935-466b-ab70-5db99be4bac5
# ╠═153c0d22-398d-40d9-a345-24ea3cd28798
# ╠═b3d350f9-0273-4071-8d56-69765f550c62
# ╟─44895f8d-3876-4c92-a7cf-38ee02189a79
# ╠═653e631e-c2d6-444b-abd3-b75638b2101a
# ╠═0f7590a0-11ee-4a2c-9236-efe494d15bc5
# ╠═bd1c3129-e1e3-45ea-8c4e-ee26e073fc09
# ╠═17c96edb-8a0b-436c-a6ed-d7dd8bc4c1c4
# ╠═0dbe68f0-420c-48ac-bd5e-8720edd1dfbf
# ╠═9310dda2-a8b1-4cb8-9663-07cbce8f80c2
# ╠═4ad7cd3f-29d9-4bab-8cd4-1d3315a29e58
# ╠═a85b659d-e95a-4194-ac0c-fb49267b99de
# ╠═e3daab95-21a5-4403-87dc-6faa67ff93a2
# ╠═9530a0b1-0922-409f-8af8-b91d186d0d2d
# ╠═57eced27-b9e4-4e12-a109-92c0897b5dee
# ╠═b223c591-ebda-4adb-86ff-b58fbfce050c
# ╟─e95f6771-984d-4f1d-ac24-f4554a84e431
# ╠═0dc1e94b-61fd-4739-9dee-38f7e24946c1
# ╠═7da23a18-7df6-4060-8070-923a752b7f84
