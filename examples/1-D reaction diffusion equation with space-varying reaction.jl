### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ c00635e2-6915-4dbf-8675-ba16645f8182
@bind N PlutoUI.Slider(0:100; default=25, show_value=N -> "Order : N=$N")

# ╔═╡ 08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
c = 3; nothing

# ╔═╡ 7636cea2-c846-4c71-bba7-7168ef006283
ε = 1; nothing

# ╔═╡ d1504ce0-6b32-488b-9392-ba824b979c8e
begin
	λ_ps = TaylorExpansionSeries{Float64}(:lambda, [x], [3 + x^2*sin(3*x)], [0])
	compute_coefficients!(λ_ps, N)
	λ = SymbolicSeries(λ_ps)
end; nothing

# ╔═╡ e683d8aa-5c46-4eaf-8ae8-045fdd6cc7a2
md"""
# PDE and boundary conditions
"""

# ╔═╡ 467f87c5-4f6d-4466-a53a-fe231f16e0d1
unknown = selfseries_symbols(); nothing

# ╔═╡ 61a0b5b2-2a67-4322-af15-46d961b21cf8
K = SymbolicSeries(unknown, [0,0]); nothing

# ╔═╡ 00de77c1-e8f0-428b-a327-6f5af696aab9
md"""
### Boundary conditions
"""

# ╔═╡ cae5baf9-4623-4feb-ac37-0455552f45c1
BC1 = K(x,0) ~ 0; nothing

# ╔═╡ b38b0c10-ffa6-4a39-8828-805bd701c393
BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, x); nothing

# ╔═╡ 18f8860c-b213-4545-a1c6-b15a67998834
md"""
### PDE
"""

# ╔═╡ 09f4e5a7-aad0-41e0-be20-7658a390835e
PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ (λ(y)+c)/ε * K(x,y); nothing

# ╔═╡ 7a1b11be-ff45-4c2f-aa61-a7dfdb80b425
md"""
# Computing the coefficients up to order N
"""

# ╔═╡ 6ffa8c0a-7b54-4466-80dd-80a553729152
K_ps = PDESeries{Float64}(:K, [x,y], [0,0], unknown, [BC1, BC2, PDE])	

# ╔═╡ ff5ec074-be51-49f6-bfdc-7ddc9350cb77
compute_coefficients!(K_ps, N)

# ╔═╡ 17743a06-2078-47f5-8fdd-e72f2c1819f1
md"""
# Analyzing the results
"""

# ╔═╡ fe260378-2e72-49e7-a665-7dbcc38b80a1
orders = [2, 4, 6, 8, 25]; nothing

# ╔═╡ 2d6ca963-1dcf-4ae4-bdcd-c63b459ec05c
y_range = 0:0.1:1; nothing

# ╔═╡ 9697adb6-9827-406e-9886-b986ebc58a8e
begin
	Ks = []
	for order in orders
		K, = build_matrix_elt(K_ps, order)
		boundary_K(y) = K(1,y)
		push!(Ks, boundary_K)
	end
end

# ╔═╡ a01cd654-0590-4566-94db-a71040163e22
fig = Figure(); nothing

# ╔═╡ 63943345-a7fc-4c06-85ea-d36ce92ba839
ax = Axis(fig[1,1]; xlabel="y", title="K(1,y) for different values of N"); nothing

# ╔═╡ d26fc1c9-bf68-44ad-8fd3-3b440d97e949
for (order, f) in zip(orders, Ks)
	lines!(ax, y_range, f.(y_range); label="N = $order")
end

# ╔═╡ 86c8e907-be8c-4824-b444-4a0080eba9a4
axislegend(ax; position=:rt); nothing

# ╔═╡ 0dc1e94b-61fd-4739-9dee-38f7e24946c1
display(fig)

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
# ╟─c00635e2-6915-4dbf-8675-ba16645f8182
# ╠═08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
# ╠═7636cea2-c846-4c71-bba7-7168ef006283
# ╠═d1504ce0-6b32-488b-9392-ba824b979c8e
# ╟─e683d8aa-5c46-4eaf-8ae8-045fdd6cc7a2
# ╠═467f87c5-4f6d-4466-a53a-fe231f16e0d1
# ╠═61a0b5b2-2a67-4322-af15-46d961b21cf8
# ╟─00de77c1-e8f0-428b-a327-6f5af696aab9
# ╠═cae5baf9-4623-4feb-ac37-0455552f45c1
# ╠═b38b0c10-ffa6-4a39-8828-805bd701c393
# ╟─18f8860c-b213-4545-a1c6-b15a67998834
# ╠═09f4e5a7-aad0-41e0-be20-7658a390835e
# ╟─7a1b11be-ff45-4c2f-aa61-a7dfdb80b425
# ╠═6ffa8c0a-7b54-4466-80dd-80a553729152
# ╠═ff5ec074-be51-49f6-bfdc-7ddc9350cb77
# ╟─17743a06-2078-47f5-8fdd-e72f2c1819f1
# ╠═fe260378-2e72-49e7-a665-7dbcc38b80a1
# ╠═2d6ca963-1dcf-4ae4-bdcd-c63b459ec05c
# ╠═9697adb6-9827-406e-9886-b986ebc58a8e
# ╠═a01cd654-0590-4566-94db-a71040163e22
# ╠═63943345-a7fc-4c06-85ea-d36ce92ba839
# ╠═d26fc1c9-bf68-44ad-8fd3-3b440d97e949
# ╠═86c8e907-be8c-4824-b444-4a0080eba9a4
# ╠═0dc1e94b-61fd-4739-9dee-38f7e24946c1
