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
N=50

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
\begin{align*} f_N(x,y) &= \frac{λ(y)+c}{ε}K_N(x,y) - λ_{N-3}(y)(K_1(x,y)-K_0(x,y)) - \\
& λ_{N-4}(y)(K_2(x,y)-K_0(x,y)) - ... - λ_0(y)(K_{N-2}(x,y)-K_{N-3}(x,y))
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

# ╔═╡ 99decbde-b72c-4cf5-ae30-a587148f50c1
"""
	λ_N(y; N)::Vector

	Computes the values of λ_n(y) where n goes from 0 to N
"""
function λ_N(y; N)::Vector
	res = [λ_ps.coefficients[1][1]]
	pow = 1
	for c in λ_ps.coefficients[1][2:N+1]
		pow *= y
		push!(res, res[end]+c*pow)
	end
	res
end

# ╔═╡ 8d9db1cd-6515-45a8-93ee-304e053f8102
"""
	f_N(x, y; N, K_N)

	Computes the value of f_N(x,y)
"""
function f_N(x, y; N, K_N, λ_full)
	λ_Ns = λ_N(y; N=N-3)
	res = 0
	coeffs_idx = 2
	pow = 1
	for i in 1:N-2
		pow *= x
		temp = 0
		for j in 1:i
			temp += K_ps.coefficients[1][coeffs_idx]*pow
			coeffs_idx += 1
			pow /= x
			pow *= y
		end
		res -= temp*λ_Ns[N-1-i]
	end
	res + λ_full(y)*K_N(x,y)
end

# ╔═╡ be6bba23-dea2-4ca3-9bae-b5c9a2972f03
"""
	compute_bounds(N)

	Computes M_λ, M_λ(N), M_f(N)
"""
function compute_bounds(N; max_range=y_range)
	
	λ_full = Symbolics.build_function(λ_ps.func[1], λ_ps.variables[1], expression=Val(false)); nothing
	
	λ_Nminus1 = build_matrix_elt(λ_ps, N-1)[1]
	λR_N(y) = λ_full(y) - λ_Nminus1(y)
	
	M_λ = maximum(abs.(λ_full.(max_range)))
	M_λN = maximum(abs.(λR_N.(max_range)))

	K_N = build_matrix_elt(K_ps, N)[1]
	
	maxi = 0
	for x in max_range
		for y in 0:step(max_range):x
			val = abs(f_N(x,y; N, K_N, λ_full))
			if maxi <  val 
				maxi = val
			end
		end
	end
	return M_λ, M_λN, maxi
end

# ╔═╡ 94e29e22-e21c-4fe3-8a8d-41a1daf4b1ff
function abs_err(N)
	M_λ, M_λN, M_fN = compute_bounds(N)
	@show M_λ
	@show M_λN
	@show M_fN
	return exp(M_λ)*(M_λN+M_fN)*cosh(M_λ)/2
end

# ╔═╡ 99cc6730-084e-4763-99a9-6edaf8710b09
function L1_err(N)
	M_λ, M_λN, M_fN = compute_bounds(N)
	return exp(M_λ)*(M_λN+M_fN)*sinh(M_λ)/2
end

# ╔═╡ f96e1503-8925-4f70-930f-d57afd2a09ac
begin
	@show abs_err(10)
	
	@show L1_err(15)
end

# ╔═╡ 8060a0e5-caf6-48f0-b172-0464cbbd6920
function loc_err(y; N)
	M_λ, M_λN, M_fN = compute_bounds(N)
	return exp(M_λ)*(M_λN+M_fN)*cosh(M_λ*y)/2
end

# ╔═╡ 12c19ebd-1669-4fd7-90a2-10ad561a4211
"""
	plot_err(λ_ps, K_ps, N, maxOrder)

	Displays the error as computed by the above majoration vs the real error

	### Input

	- `λ_ps` -- TaylorExpansionSeries representing λ
	- `K_ps` -- PDESeries representing K
	- `order` -- The order for which the error should be displayed
	- `maxOrder=N` -- The order used to represent the "full" K

	### Output

	Displays a graph
"""
function plot_err(order, maxOrder=N)
	Ms = map(y -> loc_err(y; N=order), y_range)
	
	K_full = y -> build_matrix_elt(K_ps, maxOrder)[1](1,y)
	K_part = y -> build_matrix_elt(K_ps, order)[1](1,y)

	errs = abs.(K_full.(y_range) .- K_part.(y_range))

	fig = Figure()
	ax = Axis(fig[1,1]; xlabel="y", title="approximated error versus theoretical majoration at order $order")
	lines!(ax, y_range, Ms; label="Theoretical majoration")
	lines!(ax, y_range, errs; label="numerical estimation")

	fig[1,2] = Legend(fig, ax, "Legend", framevisible=false)

	display(fig)
end

# ╔═╡ 28561d72-52d2-4c2e-9d06-bbe76b1fdf40
plot_err(20, N)

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
# ╠═99decbde-b72c-4cf5-ae30-a587148f50c1
# ╠═8d9db1cd-6515-45a8-93ee-304e053f8102
# ╠═be6bba23-dea2-4ca3-9bae-b5c9a2972f03
# ╠═94e29e22-e21c-4fe3-8a8d-41a1daf4b1ff
# ╠═99cc6730-084e-4763-99a9-6edaf8710b09
# ╠═f96e1503-8925-4f70-930f-d57afd2a09ac
# ╠═8060a0e5-caf6-48f0-b172-0464cbbd6920
# ╠═12c19ebd-1669-4fd7-90a2-10ad561a4211
# ╠═28561d72-52d2-4c2e-9d06-bbe76b1fdf40
