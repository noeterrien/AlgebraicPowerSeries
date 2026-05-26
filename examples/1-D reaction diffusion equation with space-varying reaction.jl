### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ c4c15963-07ad-4543-8b52-928616dd2f6a
begin
	import Pkg
	Pkg.add("Symbolics")
end; nothing

# ╔═╡ 6a9b9b8e-515d-432b-95c2-bbff42229160
using Symbolics

# ╔═╡ 3af727a8-eab5-41ab-bd79-27903e3c3f53
using GLMakie

# ╔═╡ be8ecdad-3bf0-49b9-a046-3539850e74a0
include("../AlgebraicPowerSeries.jl"); nothing

# ╔═╡ 776674bd-e446-4286-b6cf-643e0e9f1e5b
md"""
# Imports
"""

# ╔═╡ d5e314ff-ec78-4308-8ddf-56c5475c68d2
import Latexify

# ╔═╡ 06d7d816-5bb2-4457-97cc-7d6b72524111
md"""
# Computing coefficients
"""

# ╔═╡ 2518dedd-3ddc-4cf0-a2bd-02cbb9eb05a6
md"""
##### Defining variables and indices
"""

# ╔═╡ b69c5445-4d87-4986-bfcf-1309079bc36d
@variables x y

# ╔═╡ 2b4329bb-6088-4b2b-ba38-2e25e05845dc
@variables i j k

# ╔═╡ 3c8656ed-1fba-4d7d-bc05-bec290b2529a
md"""
##### Parameters
"""

# ╔═╡ c00635e2-6915-4dbf-8675-ba16645f8182
N = 25 # order

# ╔═╡ d1504ce0-6b32-488b-9392-ba824b979c8e
begin
	∑λᵢ = TaylorSeries{Float64}(:lambda, [x], [6 + x^2*sin(3*x)], [0])
	compute_coefficients!(∑λᵢ, N)
end

# ╔═╡ e683d8aa-5c46-4eaf-8ae8-045fdd6cc7a2
md"""
##### Kernel computation
"""

# ╔═╡ 116e569f-fa07-4f59-b78d-a6aa761c3f36
begin 
	@variables Kᵢ₀ Kᵢⱼ Kᵢ₍ⱼ₊₂₎ Kₖⱼ
	@variables λᵢ₋₁ λᵢ₋₂₋ₖ
	@variables Σⱼ₌₀ⁱKᵢⱼ B₍ᵢ₋₂₎ⱼ
end; nothing

# ╔═╡ df0df352-1c0c-49ce-9c3f-eccc53637faf
begin 
	sc_Kᵢ₀     = SeriesCoefficient(:self, Kᵢ₀, [i,0], [i], (1,))
	sc_Kᵢⱼ     = SeriesCoefficient(:self, Kᵢⱼ, [i,j], [i,j], (1,))
	sc_Kᵢ₍ⱼ₊₂₎ = SeriesCoefficient(:self, Kᵢ₍ⱼ₊₂₎, [i,j+2], [i,j], (1,))
	sc_Kₖⱼ     = SeriesCoefficient(:self, Kₖⱼ, [k,j], [k,j], (1,))

	sc_λᵢ₋₁ = SeriesCoefficient(∑λᵢ, λᵢ₋₁, [i-1], [i], (1,))
	sc_λᵢ₋₂₋ₖ = SeriesCoefficient(∑λᵢ, λᵢ₋₂₋ₖ, [i-2-k], [i,k], (1,))
end; nothing

# ╔═╡ 0b60f870-f16f-426a-a7e4-38c06d44440c
md"""
###### First relation
"""

# ╔═╡ 278cc95e-bb7c-47f6-ac93-47de6964df2e
R1 = RecurrentRelation(Kᵢ₀ ~ 0, [i], [(0,:∞)], [sc_Kᵢ₀], []); nothing

# ╔═╡ d48fce35-639a-4785-9c52-63e961330c05
md"""
###### Second relation
"""

# ╔═╡ 297c70be-ce17-4cac-8dfd-27dc4c06a03e
sum(formulae::Vector) = +(formulae...)

# ╔═╡ 4fac33d3-c70a-4d4f-bb9c-1d2b35b476f5
ef_Σⱼ₌₀ⁱKᵢⱼ = ExpandableFormula(:Σⱼ₌₀ⁱKᵢⱼ, Σⱼ₌₀ⁱKᵢⱼ, Kᵢⱼ, [i], [j], [(0,i)], [], [sc_Kᵢⱼ], sum); nothing

# ╔═╡ 11fb4468-c439-40d0-99ff-1445a0b10094
R2 = RecurrentRelation(Σⱼ₌₀ⁱKᵢⱼ ~ -λᵢ₋₁/(2*i), [i], [(1,:∞)], [sc_λᵢ₋₁], [ef_Σⱼ₌₀ⁱKᵢⱼ]); nothing


# ╔═╡ bd3f2ffb-a7c4-409c-ab0d-eaaee537ce24
md"""
###### Third relation
"""

# ╔═╡ db4c3485-9a1d-4281-ae55-d681b9599c1f
ef_B₍ᵢ₋₂₎ⱼ = ExpandableFormula(:B₍ᵢ₋₂₎ⱼ , B₍ᵢ₋₂₎ⱼ, Kₖⱼ*λᵢ₋₂₋ₖ, [i,j], [k], [(j,i-2)], [], [sc_Kₖⱼ,sc_λᵢ₋₂₋ₖ], sum); nothing

# ╔═╡ 4a420f7a-3008-42a1-a30e-b9ce0f858fad
R3 = RecurrentRelation((i-j)*(i-j-1)*Kᵢⱼ - (j+2)*(j+1)*Kᵢ₍ⱼ₊₂₎ ~ B₍ᵢ₋₂₎ⱼ, [i,j], 
                       [(2,:∞),(0,i-2)], [sc_Kᵢⱼ, sc_Kᵢ₍ⱼ₊₂₎], [ef_B₍ᵢ₋₂₎ⱼ]); nothing

# ╔═╡ 679c481c-0697-4d35-992b-531a68a6d694
md"""
###### Series definition and computation
"""

# ╔═╡ 1db49b0a-ecae-4d99-bd2d-4abec3fe79a6
begin
	rs = RecurrentSeries{Float64}(:K, (1,), [x,y], [0,0], [R1, R2, R3])
	compute_coefficients!(rs, N)
end

# ╔═╡ 17743a06-2078-47f5-8fdd-e72f2c1819f1
md"""
# Study of the series' convergence
"""

# ╔═╡ cd753313-5142-4024-9a12-0ef5bfb07a9c
md"""
##### Create function
"""

# ╔═╡ 4d8d1fc9-748d-4841-80fe-de23b7976c35
orders = [2,4,6,8,25]

# ╔═╡ dd8dc718-9aa4-4cc4-ad4a-9b979ad4eae2
funcs = [build_matrix_elt(rs, o)[1] for o in orders]; nothing

# ╔═╡ 13f578f8-0097-48c0-8c10-62fc1e20b6ed
x_values = [0; 0.5;; 0.75; 1]

# ╔═╡ 41ad664a-2a31-400b-a4f2-5e71296b2beb
bound_funcs = map(ω -> [μ -> func(ω,μ) for func in funcs], x_values); nothing

# ╔═╡ 004bea75-8225-4a26-bf7a-f583e6b00b56
md"""
##### Plotting
"""

# ╔═╡ 0a633733-76cb-4363-b04a-6c0a9f660873
md"""
###### 2D
"""

# ╔═╡ 64d13d07-7a9f-4ab9-84ce-e9747222617e
begin
	f = Figure()
	axs = Matrix{Axis}(undef, 2, 2)
	for (i, x_value) in zip(eachindex(axs), x_values)
		axs[i] = Axis(f[(i-1)÷2+1, (i-1)%2+1]; xlabel="y", title="K($x_value,y) for different values of the order N")
	end
end

# ╔═╡ 2847c01b-026b-40fa-92ae-c2f14ec52fd4
range = 0:0.05:1

# ╔═╡ d7062d44-f9b3-40e9-a28a-dc905f97eea4
for i in eachindex(axs)
	for (o,func) in zip(orders, bound_funcs[i])
		lines!(axs[i], range, func.(range); label="N=$o")
	end
end

# ╔═╡ d8e588a0-547c-4f24-8b09-713d15175314
for ax in axs
	axislegend(ax; position=:rt)
end

# ╔═╡ 14db96a9-f5d6-41fe-b747-9c730971a6a5
display(f)

# ╔═╡ 0074b6c6-c236-4d48-bff4-477cf0b8a350
md"""
###### 3D
"""

# ╔═╡ 3faabdbc-5f89-4d4f-9e8c-facd5398793d
display(surface(range, range, [funcs[1](x,y) for x in range, y in range], axis=(type=Axis3,title="K(x,y) for N=2", xlabel="x", ylabel="y", zlabel="K(x,y)")))

# ╔═╡ bfa7aef2-24fb-429d-b648-2234f8547656
display(surface(range, range, [funcs[end-1](x,y) for x in range, y in range], axis=(type=Axis3,title="K(x,y) for N=8", xlabel="x", ylabel="y", zlabel="K(x,y)")))

# ╔═╡ df3b9147-ee7a-4638-a9d8-8b18fcb565b9
display(surface(range, range, [funcs[end](x,y) for x in range, y in range], axis=(type=Axis3,title="K(x,y) for N=25", xlabel="x", ylabel="y", zlabel="K(x,y)")))

# ╔═╡ Cell order:
# ╟─776674bd-e446-4286-b6cf-643e0e9f1e5b
# ╠═c4c15963-07ad-4543-8b52-928616dd2f6a
# ╠═6a9b9b8e-515d-432b-95c2-bbff42229160
# ╠═be8ecdad-3bf0-49b9-a046-3539850e74a0
# ╠═d5e314ff-ec78-4308-8ddf-56c5475c68d2
# ╠═3af727a8-eab5-41ab-bd79-27903e3c3f53
# ╟─06d7d816-5bb2-4457-97cc-7d6b72524111
# ╟─2518dedd-3ddc-4cf0-a2bd-02cbb9eb05a6
# ╠═b69c5445-4d87-4986-bfcf-1309079bc36d
# ╠═2b4329bb-6088-4b2b-ba38-2e25e05845dc
# ╟─3c8656ed-1fba-4d7d-bc05-bec290b2529a
# ╠═c00635e2-6915-4dbf-8675-ba16645f8182
# ╠═d1504ce0-6b32-488b-9392-ba824b979c8e
# ╟─e683d8aa-5c46-4eaf-8ae8-045fdd6cc7a2
# ╠═116e569f-fa07-4f59-b78d-a6aa761c3f36
# ╠═df0df352-1c0c-49ce-9c3f-eccc53637faf
# ╟─0b60f870-f16f-426a-a7e4-38c06d44440c
# ╠═278cc95e-bb7c-47f6-ac93-47de6964df2e
# ╟─d48fce35-639a-4785-9c52-63e961330c05
# ╠═297c70be-ce17-4cac-8dfd-27dc4c06a03e
# ╠═4fac33d3-c70a-4d4f-bb9c-1d2b35b476f5
# ╠═11fb4468-c439-40d0-99ff-1445a0b10094
# ╟─bd3f2ffb-a7c4-409c-ab0d-eaaee537ce24
# ╠═db4c3485-9a1d-4281-ae55-d681b9599c1f
# ╠═4a420f7a-3008-42a1-a30e-b9ce0f858fad
# ╟─679c481c-0697-4d35-992b-531a68a6d694
# ╠═1db49b0a-ecae-4d99-bd2d-4abec3fe79a6
# ╟─17743a06-2078-47f5-8fdd-e72f2c1819f1
# ╟─cd753313-5142-4024-9a12-0ef5bfb07a9c
# ╠═4d8d1fc9-748d-4841-80fe-de23b7976c35
# ╠═dd8dc718-9aa4-4cc4-ad4a-9b979ad4eae2
# ╠═13f578f8-0097-48c0-8c10-62fc1e20b6ed
# ╠═41ad664a-2a31-400b-a4f2-5e71296b2beb
# ╟─004bea75-8225-4a26-bf7a-f583e6b00b56
# ╟─0a633733-76cb-4363-b04a-6c0a9f660873
# ╠═64d13d07-7a9f-4ab9-84ce-e9747222617e
# ╠═2847c01b-026b-40fa-92ae-c2f14ec52fd4
# ╠═d7062d44-f9b3-40e9-a28a-dc905f97eea4
# ╠═d8e588a0-547c-4f24-8b09-713d15175314
# ╠═14db96a9-f5d6-41fe-b747-9c730971a6a5
# ╟─0074b6c6-c236-4d48-bff4-477cf0b8a350
# ╠═3faabdbc-5f89-4d4f-9e8c-facd5398793d
# ╠═bfa7aef2-24fb-429d-b648-2234f8547656
# ╠═df3b9147-ee7a-4638-a9d8-8b18fcb565b9
