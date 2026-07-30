### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ c00635e2-6915-4dbf-8675-ba16645f8182
N=100

# ╔═╡ 08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
c = 3; nothing

# ╔═╡ 7636cea2-c846-4c71-bba7-7168ef006283
ε = 1; nothing

# ╔═╡ 3f5a2d98-574e-4380-8aa3-9726c087cfdb
λ_full(x) = 3+x^2*sin(3*x)

# ╔═╡ d1504ce0-6b32-488b-9392-ba824b979c8e
begin
	λ_ps = TaylorExpansionSeries{Float64}(:lambda, [x], [λ_full(x)], [0])
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
BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x)+c, x); nothing

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
y_range = 0:0.01:1; nothing

# ╔═╡ 9697adb6-9827-406e-9886-b986ebc58a8e
begin
	Ks = []
	for order in orders
		local K, = build_matrix_elt(K_ps, order)
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

# ╔═╡ d113dd59-314c-4173-a23e-2c6fa610626b
md"""
# Computing the error
"""

# ╔═╡ 0a4535ec-f9da-469d-a3e7-902ad72aaab5
md"""
We are trying to find the error on the rest when solving the system of equations

```math
\begin{align*}
K_{xx}(x,y) - K_{yy}(x,y) &= \frac{λ(ξ) + c}{ε}*K(x,y) \\
K(x,x) &= -\frac{1}{2ε} \int_0^x (λ(y) + c) dy \\ 
K(x,0) &= 0
\end{align*}
```

using power series. For that, we denote 

``
K_N(x,y) = \sum_{i=0}^N \sum_{j=0}^i K_{ij} x^iy^j
`` and 
``
R_N(x,y) = K(x,y) - K_N(x,y) = \sum_{i=N+1}^{+∞} \sum_{j=0}^{i} K_{ij} x^iy^j
`` for all ``N ∈ \mathbb{N}``

We also denote 

``\frac{λ(y)+c}{ε} = \sum_{j=0}^{+∞} λ_jx^j, λ_N(y) = \sum_{j=0}^N λ_jx^j`` and ``λR_N(y) = \sum_{j=N+1}^{+∞} λ_jx^j``
"""

# ╔═╡ 00fa41cb-435a-4070-a234-57d6aaf1106b
md"""
We define ``f_N`` as 

```math
\begin{align*} f_N(x,y) &= λR_N(y)K_N(x,y) + \sum_{j=0}^N(\sum_{k=0}^j K_{(N-j+k)k}λ_{j-k}))x^{N-j}y^j + \sum_{j=0}^{N-1}(\sum_{k=0}^j K_{(N-1-j+k)k}λ_{j-k}))x^{N-1-j}y^j
\end{align*}
```
"""

# ╔═╡ e14afcf4-4782-4048-8c5a-5340fbe895cd
md"""
Developping the differential equation on ``R_N``, one gets that it must be solution to

```math
\begin{align}
R_{N,xx}-R_{N,yy} &= λ(y)R_N(x,y) + f_N(x,y) \\
R_N(x,x) &= -\frac{1}{2ε} \int_0^x λR_{N-1}(y) dy \\
R_N(x,0) = 0
\end{align}
```

and using the method developped in sections II-A and II-B of "The power series method to compute backstepping kernel gains: theory and practice,
Rafael Vazquez, Guangwei Chen, Junfei Qiao and Miroslav Krstic", it is possible to show that


If ``∀ (x,y) \in T = \left\{(x,y) \in R^2 | 0\leq y \leq x \leq 1\right\}``, 
```math 
\begin{align*}
|\frac{λ(y)+c}{ε}| &\leq M_{λ} \\
|λR_N(y)| &\leq M_{λ}^{(N)} \\
|f_N(x,y)| &\leq M_f^{(N)}
\end{align*}
```

Then 
```math
∀ y \in [0,1], |R_N(1,y)| \leq e^{M_λ}\frac{M_λ^{(N-1)}+M_f^{(N)}}{2}cosh(M_λy)
```

This is the inequality implemented here
"""

# ╔═╡ 163bc010-1a3b-40e0-8094-c5be51de8fa7
function generate_f_N(N)
	λ_N = build_matrix_elt(λ_ps, N)[1]
	λR_N(y) = λ_full(y) - λ_N(y)
	K_N = build_matrix_elt(K_ps, N)[1]
	function f_N(x,y)
		
		res = λR_N(y)*K_N(x,y)
		
		pow_x = x^N
		pow_y = 1
		for j in 0:N
			temp = 0
			for k in 0:j
				K_coeff = K_ps.coefficients[1][dynamic_convertIndices_trunc_to_lin(N-j+k, k)]
				λ_coeff = λ_ps.coefficients[1][j-k+1]
				temp += K_coeff*λ_coeff
			end
			res += temp*pow_x*pow_y
			pow_x = x == 0 ? 0 : pow_x/x
			pow_y *= y
		end

		pow_x = x^(N-1)
		pow_y = 1
		for j in 0:N-1
			temp = 0
			for k in 0:j
				K_coeff = K_ps.coefficients[1][dynamic_convertIndices_trunc_to_lin(N-1-j+k, k)]
				λ_coeff = λ_ps.coefficients[1][j-k+1]
				temp += K_coeff*λ_coeff
			end
			res += temp*pow_x*pow_y
			pow_x = x == 0 ? 0 : pow_x/x
			pow_y *= y
		end

		return res
	end
end

# ╔═╡ eb303137-4670-4ee7-b7d7-319641a44d28
function M_f(N; step=0.01)
	f_N = generate_f_N(N)
	M = abs(f_N(0,0))
	for x in 0:step:1
		for y in 0:step:x
			M = max(M, abs(f_N(x,y)))
		end
	end
	return M
end

# ╔═╡ 3a560ae8-5b57-4375-a0a1-afc87aeb8dd3
begin
	M_λ = maximum(map(λ_full, 0:0.01:1))
	function M_λR(N; step=0.01)
		λ_N = build_matrix_elt(λ_ps, N)[1]
		λR_N(y) = λ_full(y) - λ_N(y)
		maximum(map(λR_N, 0:step:1))
	end
end


# ╔═╡ 7a48b90f-9102-42d3-8cb1-a3d379a53143
function plot_errs(rg; num_estim_N, step=0.01)
	K_Nmax = build_matrix_elt(K_ps, num_estim_N)[1]
	fig = Figure()
	ax = Axis(fig[1,1]; title="numerical error vs theoretical error", xlabel="N", yscale=log10)
	num_errs = []
	theo_errs = []
	for N in rg
		# numerical error
		K_N = build_matrix_elt(K_ps, N)[1]
		push!(num_errs, abs(K_Nmax(0,0)-K_N(0,0)))
		for x in 0:step:1
			for y in 0:step:x
				num_errs[end] = max(num_errs[end], abs(K_Nmax(x,y)-K_N(x,y)))
			end
		end

		# theoretical error
		push!(theo_errs, exp(M_λ)*(M_λR(N-1; step)+M_f(N; step))/2*cosh(M_λ))

	end
	scatterlines!(rg, num_errs; label="numerical errors")
	scatterlines!(rg, theo_errs; label="theoretical bound")
	fig, ax
end

# ╔═╡ c297015d-0c77-4267-b196-ea0df27fab86
err_fig, err_ax = plot_errs(1:100; num_estim_N=100)

# ╔═╡ c67b8b71-2ea4-40f1-9046-5f41a8588acf
vlines!(err_ax, [80]; color=:red)

# ╔═╡ a282b63d-459b-4a00-be7c-d42ee57b5ea3
Legend(err_fig[1,2], err_ax)

# ╔═╡ 988ad4f4-69cb-4c79-a3ae-d7e023d09b35
display(err_fig)

# ╔═╡ 2548d88b-d904-4563-ae72-19c5dc6fb290
save("numerical error vs theoretical error.png", err_fig)

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
# ╠═c00635e2-6915-4dbf-8675-ba16645f8182
# ╠═08f7c3f2-fcd1-4d16-bd24-3dbe928c334e
# ╠═7636cea2-c846-4c71-bba7-7168ef006283
# ╠═3f5a2d98-574e-4380-8aa3-9726c087cfdb
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
# ╟─d113dd59-314c-4173-a23e-2c6fa610626b
# ╟─0a4535ec-f9da-469d-a3e7-902ad72aaab5
# ╟─00fa41cb-435a-4070-a234-57d6aaf1106b
# ╟─e14afcf4-4782-4048-8c5a-5340fbe895cd
# ╠═163bc010-1a3b-40e0-8094-c5be51de8fa7
# ╠═eb303137-4670-4ee7-b7d7-319641a44d28
# ╠═3a560ae8-5b57-4375-a0a1-afc87aeb8dd3
# ╠═7a48b90f-9102-42d3-8cb1-a3d379a53143
# ╠═c297015d-0c77-4267-b196-ea0df27fab86
# ╠═c67b8b71-2ea4-40f1-9046-5f41a8588acf
# ╠═a282b63d-459b-4a00-be7c-d42ee57b5ea3
# ╠═988ad4f4-69cb-4c79-a3ae-d7e023d09b35
# ╠═2548d88b-d904-4563-ae72-19c5dc6fb290
