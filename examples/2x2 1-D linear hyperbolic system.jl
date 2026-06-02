### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
begin
	import Pkg
	Pkg.add("Symbolics")
end; nothing

# ╔═╡ 5ad1546c-1481-42e2-8b99-db808eea7768
using Symbolics

# ╔═╡ 34c18021-d9bf-42d8-a9ba-001682b2cee5
using GLMakie

# ╔═╡ 4c1068dd-466a-4b9f-b37f-14c30e381571
include("../AlgebraicPowerSeries.jl"); nothing

# ╔═╡ b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
md"""
# Imports
"""

# ╔═╡ 9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
import Latexify

# ╔═╡ 99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
md"""
# Computing coefficients
"""

# ╔═╡ 089778d9-8560-49b8-8c3a-853df5536cfb
md"""
##### Defining variables and indices
"""

# ╔═╡ 838b2173-6a55-4eb1-8a08-7cd98552d780
@variables x y

# ╔═╡ 23cfd29e-4cd9-49a0-ba93-338ed60b4e57
@variables i j k

# ╔═╡ 902b946a-a877-42b3-8c18-81a4fc9a4262
md"""
##### Parameters
"""

# ╔═╡ fa0e5bce-5ba3-4833-aed5-545e31d166de
N = 30 # order

# ╔═╡ c26326ea-ad12-4f6c-9dac-40bb68133c7f
q = 1

# ╔═╡ be9ce343-7f72-48e6-8661-0a6a4c2c77f1
md"""
##### Σ function
"""

# ╔═╡ def0ec8e-e807-469d-b10f-425b56720a86
begin
	Σ = TaylorExpansionSeries{Float64}(:Σ, [x], [1.2+x^3;0;;0;1.5+x^2], [0])
	compute_coefficients!(Σ, N+1); 
end; nothing

# ╔═╡ c47fa1af-af58-4b85-84ae-3d01c64a0b4a
md"""
##### C function
"""

# ╔═╡ 9cd56389-61d8-4641-828a-b46180e8e891
begin
	C = TaylorExpansionSeries{Float64}(:C, [x], [3*cos(x);1+2*exp(x);;sin(2*x);1/(3+x^2)], [0])
	compute_coefficients!(C, N)
end; nothing

# ╔═╡ 0527f200-18c8-4098-b3ce-9dfaf7fb92ae
md"""
##### kernel representation
"""

# ╔═╡ 7cb2c546-9353-49a4-a14a-413d8c8dbdbe
K = selfseries_symbols(2,2); nothing

# ╔═╡ 73949d4a-9e28-4045-8977-c45fc17329ad
md"""
##### Relations of recurrence
"""

# ╔═╡ 2339fa53-55c3-47a5-8e67-4bcfe54ed54d
md"""
###### First relation
"""

# ╔═╡ 89fab97f-7a50-4319-aa4b-27af5a1d6ae5
begin
	R11 = @recurrent_relation K[1,1][i,0] ~ Σ[2,2][0]/(q*Σ[1,1][0])*K[1,2][i,0] i in 0:(:∞)
	R12 = @recurrent_relation K[2,1][i,0] ~ Σ[2,2][0]/(q*Σ[1,1][0])*K[2,2][i,0] i in 0:(:∞)
end; nothing

# ╔═╡ 7201eee7-96a2-4d9f-ae5d-3326017be86c
md"""
###### Second relation
"""

# ╔═╡ 0d8b4e73-b99c-4ecd-a38a-3005450aa69c
begin
	R21 = @recurrent_relation (@∑ K[1,2][k,j]*(Σ[1,1][i-k]+Σ[2,2][i-k]) j in 0:k k in 0:i) ~  C[1,2][i] i in 0:(:∞)
	R22 = @recurrent_relation (@∑ K[2,1][k,j]*(Σ[1,1][i-k]+Σ[2,2][i-k]) j in 0:k k in 0:i) ~ -C[2,1][i] i in 0:(:∞)
end; nothing

# ╔═╡ 2f923e1d-7945-4ce4-a785-d86fa3b272ed
md"""
###### Third relation
"""

# ╔═╡ 5894eeff-f2eb-4f3a-8976-5fb74a145fc0
begin
	R31 = @recurrent_relation(
			  (@∑ (k+1)*K[1,1][k+j+1,j]*Σ[1,1][i-1-j-k] k in 0:(i-1-j)) 
			+ (@∑ (k+1)*K[1,1][i-1-j+k+1,k+1]*Σ[1,1][j-k] k in 0:j) ~   
			- (@∑ (j-k+1)*K[1,1][i-1-j+k,k]*Σ[1,1][j-k+1] k in 0:j) 
			- (@∑ K[1,2][i-1-j+k,k]*C[2,1][j-k] k in 0:j) 
			+ (@∑ K[1,1][k+j,j]*C[1,1][i-1-j-k] k in 0:(i-1-j))
			- (@∑ K[1,1][i-1-j+k,k]*C[1,1][j-k] k in 0:j),
			j in 0:(i-1), i in 0:(:∞)
		)
	R32 = @recurrent_relation(
		      (@∑ (k+1)*K[1,2][k+j+1,j]*Σ[1,1][i-1-j-k] k in 0:(i-1-j))
		    - (@∑ (k+1)*K[1,2][i-1-j+k+1,k+1]*Σ[2,2][j-k] k in 0:j) ~  
			  (@∑ (j-k+1)*K[1,2][i-1-j+k,k]*Σ[2,2][j-k+1] k in 0:j) 
			- (@∑ K[1,1][i-1-j+k,k]*C[1,2][j-k] k in 0:j) 
			+ (@∑ K[1,2][k+j,j]*C[1,1][i-1-j-k] k in 0:(i-1-j))
			- (@∑ K[1,2][i-1-j+k,k]*C[2,2][j-k] k in 0:j),
			j in 0:(i-1), i in 0:(:∞)
		)
	R33 = @recurrent_relation(
			  (@∑ (k+1)*K[2,1][k+j+1,j]*Σ[2,2][i-1-j-k] k in 0:(i-1-j)) 
			- (@∑ (k+1)*K[2,1][i-1-j+k+1,k+1]*Σ[1,1][j-k] k in 0:j) ~  
			  (@∑ (j-k+1)*K[2,1][i-1-j+k,k]*Σ[1,1][j-k+1] k in 0:j) 
			+ (@∑ K[2,2][i-1-j+k,k]*C[2,1][j-k] k in 0:j) 
			- (@∑ K[2,1][k+j,j]*C[2,2][i-1-j-k] k in 0:(i-1-j))
			+ (@∑ K[2,1][i-1-j+k,k]*C[1,1][j-k] k in 0:j),
			j in 0:(i-1), i in 0:(:∞)
		) 
	R34 = @recurrent_relation(
			  (@∑ (k+1)*K[2,2][k+j+1,j]*Σ[2,2][i-1-j-k] k in 0:(i-1-j)) 
			+ (@∑ (k+1)*K[2,2][i-1-j+k+1,k+1]*Σ[2,2][j-k] k in 0:j) ~ 
			- (@∑ (j-k+1)*K[2,2][i-1-j+k,k]*Σ[2,2][j-k+1] k in 0:j) 
			+ (@∑ K[2,1][i-1-j+k,k]*C[1,2][j-k] k in 0:j) 
			- (@∑ K[2,2][k+j,j]*C[2,2][i-1-j-k] k in 0:(i-1-j))
			+ (@∑ K[2,2][i-1-j+k,k]*C[2,2][j-k] k in 0:j),
			j in 0:(i-1), i in 0:(:∞)
		)
end; nothing

# ╔═╡ ebf696cb-3e04-4ead-b5fa-23c0991a8eb1
md"""
###### Series definition and computation
"""

# ╔═╡ 8009dbf2-4d50-4640-a22b-8c62cd841be3
begin
	rs = RecurrentSeries{Float64}(:crs, (2,2), [x,y], [0,0], [R11, R12, R21, R22, R31, R32, R33, R34])
	compute_coefficients!(rs, N)
end

# ╔═╡ c30e52dd-444e-43a3-bbb3-2b25b63311af
md"""
# Study of the series' convergence
"""

# ╔═╡ c191ce94-7a8a-4791-8449-cba849cc74ed
orders = [15, 20, 25, 30]

# ╔═╡ 4a36eaa3-93b9-4f58-a522-a17ea2566801
funcs = [ω -> build(rs, o)(1, ω) for o in orders]; nothing

# ╔═╡ d6e8bddc-851c-4195-994e-c8511ed27d82
begin 
	fig = Figure()
	axs = Matrix{Axis}(undef, 2,2)
	labels = ["Kᵘᵘ"; "Kᵛᵘ";; "Kᵘᵛ"; "Kᵛᵛ"]
	for i in axes(axs, 1), j in axes(axs, 2)
		axs[i,j] = Axis(fig[i,j]; xlabel="x", title="$(labels[i,j])(1,y)")
	end
end

# ╔═╡ f325fa04-1623-4a9d-9ad2-aed7557c7f4d
range = 0:0.05:1

# ╔═╡ 45b0e870-9d85-486e-841b-0dfc0da5655c
begin
	data = []
	for func in funcs
		push!(data, stack(func.(range), dims=3))
	end
end

# ╔═╡ 85fecc9a-a450-49c3-b426-d83d965d26ef
for (d,order) in zip(data,orders)
	for i in axes(axs, 1), j in axes(axs, 2)
		lines!(axs[i,j], range, d[i,j,:]; label="N=$order")
	end
end

# ╔═╡ f183a7c4-baaa-494f-9c9a-d944617b273d
foreach(ax -> axislegend(ax; position=:rt), axs)

# ╔═╡ 12b0551e-c67a-485b-80e0-6d00e91bf853
fig

# ╔═╡ Cell order:
# ╟─b34ceca0-58f9-11f1-9c39-fd7d92bf34d9
# ╠═c676d3ac-ef7b-47f2-b4ac-99f56ddd93d8
# ╠═5ad1546c-1481-42e2-8b99-db808eea7768
# ╠═4c1068dd-466a-4b9f-b37f-14c30e381571
# ╠═9df7d598-6c81-4fa4-97e2-fe031e0ae1d5
# ╠═34c18021-d9bf-42d8-a9ba-001682b2cee5
# ╟─99f61bb3-47ef-49e0-8e4e-e25bed05c5c6
# ╟─089778d9-8560-49b8-8c3a-853df5536cfb
# ╠═838b2173-6a55-4eb1-8a08-7cd98552d780
# ╠═23cfd29e-4cd9-49a0-ba93-338ed60b4e57
# ╟─902b946a-a877-42b3-8c18-81a4fc9a4262
# ╠═fa0e5bce-5ba3-4833-aed5-545e31d166de
# ╠═c26326ea-ad12-4f6c-9dac-40bb68133c7f
# ╟─be9ce343-7f72-48e6-8661-0a6a4c2c77f1
# ╠═def0ec8e-e807-469d-b10f-425b56720a86
# ╟─c47fa1af-af58-4b85-84ae-3d01c64a0b4a
# ╠═9cd56389-61d8-4641-828a-b46180e8e891
# ╟─0527f200-18c8-4098-b3ce-9dfaf7fb92ae
# ╠═7cb2c546-9353-49a4-a14a-413d8c8dbdbe
# ╟─73949d4a-9e28-4045-8977-c45fc17329ad
# ╟─2339fa53-55c3-47a5-8e67-4bcfe54ed54d
# ╠═89fab97f-7a50-4319-aa4b-27af5a1d6ae5
# ╟─7201eee7-96a2-4d9f-ae5d-3326017be86c
# ╠═0d8b4e73-b99c-4ecd-a38a-3005450aa69c
# ╟─2f923e1d-7945-4ce4-a785-d86fa3b272ed
# ╠═5894eeff-f2eb-4f3a-8976-5fb74a145fc0
# ╟─ebf696cb-3e04-4ead-b5fa-23c0991a8eb1
# ╠═8009dbf2-4d50-4640-a22b-8c62cd841be3
# ╟─c30e52dd-444e-43a3-bbb3-2b25b63311af
# ╠═c191ce94-7a8a-4791-8449-cba849cc74ed
# ╠═4a36eaa3-93b9-4f58-a522-a17ea2566801
# ╠═d6e8bddc-851c-4195-994e-c8511ed27d82
# ╠═f325fa04-1623-4a9d-9ad2-aed7557c7f4d
# ╠═45b0e870-9d85-486e-841b-0dfc0da5655c
# ╠═85fecc9a-a450-49c3-b426-d83d965d26ef
# ╠═f183a7c4-baaa-494f-9c9a-d944617b273d
# ╠═12b0551e-c67a-485b-80e0-6d00e91bf853
