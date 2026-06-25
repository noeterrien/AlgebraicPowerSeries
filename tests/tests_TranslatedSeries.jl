using Test
using BenchmarkTools

include("../AlgebraicPowerSeries.jl")

@variables x y z t

sin_ps = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
tr_sin = TranslatedSeries(:tr_sin, sin_ps, [1])
compute_coefficients!(tr_sin, 10; trunc_order=30)

tr_sin_ps = TaylorExpansionSeries{Float64}(:tr_sin_ps, [x], [sin(x)], [1])
compute_coefficients!(tr_sin_ps, 10)
@test all(tr_sin.coefficients[1] .≈ tr_sin_ps.coefficients[1])


cosin_tes_0 = TaylorExpansionSeries{Float64}(:cosin_0, [x,y], [cos(x+y)*sin(x-y); cos(x);; sin(y); sin(x+y)*cos(x-y)], [0,0])
cosin_trs_1 = TranslatedSeries(:cosin_tr_1, cosin_tes_0, [1,1])
compute_coefficients!(cosin_trs_1, 10; trunc_order=20)

cosin_tes_1 = TaylorExpansionSeries{Float64}(:cosin_1, [x,y], [cos(x+y)*sin(x-y); cos(x);; sin(y); sin(x+y)*cos(x-y)], [1,1])
compute_coefficients!(cosin_tes_1, 10)
tol = 1e-8
@test all(.≈(cosin_trs_1.coefficients[1,1], cosin_tes_1.coefficients[1,1], atol=tol))
@test all(.≈(cosin_trs_1.coefficients[1,2], cosin_tes_1.coefficients[1,2], atol=tol))
@test all(.≈(cosin_trs_1.coefficients[2,1], cosin_tes_1.coefficients[2,1], atol=tol))
@test all(.≈(cosin_trs_1.coefficients[2,2], cosin_tes_1.coefficients[2,2], atol=tol))