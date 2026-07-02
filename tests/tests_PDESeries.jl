include("../AlgebraicPowerSeries.jl")

using Test

@variables x y z t
@variables i j k

sin_series = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
K = selfseries_symbols(2,2)

sin_ss = SymbolicSeries(sin_series)
K_ss = SymbolicSeries(K, [0])

getNum((K_ss[1,1] + K_ss[1,2] + K_ss[2,1]) * 4, 2)
getNum(K_ss[1,1] / 3, 1)
getNum(x*K_ss[2,2], 4)


@test getSymbolics(sin_ss*sin_ss, 4) ≈ getSymbolics((sin_ss*sin_ss)[4])

sincos_series = TaylorExpansionSeries{Float64}(:sincos, [x,y], [sin(x), cos(y)], [0,0])
compute_coefficients!(sincos_series, 6)
sincos_ss = SymbolicSeries(sincos_series)
@test getNum(sincos_ss[1][1,1]) ≈ 0
@test getNum(sincos_ss[2][0,2]) ≈ -1/2

sincos_ss[1](x,y)/3

for i in 0:3, j in 0:3, k in 0:3
    if (i,j,k) != (1,0,0) && (i,j,k) != (3,0,0)
        @test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[i,j,k]) ≈ 0
    end
end
@test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[1,0,0]) ≈ 1
@test getNum((sincos_ss[1](y,z)*sincos_ss[2](t,y))[3,0,0]) ≈ -2/3

∂x = Differential(x)
@test getNum(∂x(sincos_ss[1](x,y))[0,0]) ≈ 1
@test getNum(∂x(sincos_ss[1](x,y))[2,0]) ≈ -1/2

test1_ts = TaylorExpansionSeries{Float64}(:t1, [x,y,z], [x^2 + 2x*y + y^2 + x + z + 3y*z], [0,0,0])
compute_coefficients!(test1_ts, 10)
test1_ss = SymbolicSeries(test1_ts)


@test getNum(test1_ss(x, x, z)[2,0]) ≈ 4
@test getNum(test1_ss(x, y, y)[0,2]) ≈ 4
@test getNum(test1_ss(x, z, z)[0,1]) ≈ 1
@test getNum(test1_ss(x, y, x)[1,0]) ≈ 2
@test getNum(test1_ss(x, y, x)[1,1]) ≈ 5

@test (K_ss[1,1]*K_ss[1,2]).get_selfseries_coefficients(2) == Set([K_ss[1,1][0], K_ss[1,1][1], K_ss[1,1][2],
                                                                  K_ss[1,2][0], K_ss[1,2][1], K_ss[1,2][2]])

c = SymbolicSeries(TaylorExpansionSeries{Float64}(:c, [x], [1/(3+x^2)], [0]))
K = SymbolicSeries(selfseries_symbols(2), [0,0])
@show getSymbolics((c(x)*K[1](x,y))[1,0])
@show getSymbolics((c(x)*K[1](x,y))[0,1])

@test getNum(test1_ss(x,2,3)[0,N=2]) ≈ 25
@test getNum(test1_ss(x,2,3)[1,N=2]) ≈ 5
@test getNum(test1_ss(x,2,3)[2,N=2]) ≈ 1

@test getNum(test1_ss(1,y,y)[0,N=2]) ≈ 2
@test getNum(test1_ss(1,y,y)[1,N=2]) ≈ 3
@test getNum(test1_ss(1,y,y)[2,N=2]) ≈ 4

test2_ts = TaylorExpansionSeries{Float64}(:t1, [x,y,z], [(x-1)^2 + 2(x-1)*y + y^2 + x-1 + z-2 + 3y*(z-2)], [1,0,2])
compute_coefficients!(test2_ts, 10)
test2_ss = SymbolicSeries(test2_ts)

@test getNum(test2_ss(0,0,0)[N=2]) ≈ -2
@test getNum(test2_ss(x,x,x)[0,N=2]) ≈ -3
@test getNum(test2_ss(x,x,x)[1,N=2]) ≈ 6
@test getNum(test2_ss(x,x,x)[2,N=2]) ≈ 7

@test getNum(translate(test2_ss, [0,0,0])(0,0,0)[N=2]) ≈ -2
@test getNum(translate(test2_ss, [0,0,0])(x,x,x)[0,N=2]) ≈ -2
@test getNum(translate(test2_ss, [0,0,0])(x,x,x)[1,N=2]) ≈ -8
@test getNum(translate(test2_ss, [0,0,0])(x,x,x)[2,N=2]) ≈ 7



#-----------------------------------------------------Real use-case testing-----------------------------------------------
@variables x ξ
∂x, ∂ξ = Differential(x), Differential(ξ)

N = 3 #order
q = 1

μ_ts = TaylorExpansionSeries{Float64}(:μ, [x], [1.5+x^2], [0])
compute_coefficients!(μ_ts, N+1)
μ = SymbolicSeries(μ_ts)

ϵ_ts = TaylorExpansionSeries{Float64}(:ϵ, [x], [1.2+x^3], [0])
compute_coefficients!(ϵ_ts, N+1)
ϵ = SymbolicSeries(ϵ_ts)

C_ts = TaylorExpansionSeries{Float64}(:C, [x], [3*cos(3*x);1+2*cos(2*x);;
                                                sin(2*x)  ;1/(3+x^2)    ], [0])
compute_coefficients!(C_ts, N)
C = SymbolicSeries(C_ts)

let 

    K_symbols = selfseries_symbols(2)
    K_ss = SymbolicSeries(K_symbols, [0,0])

    # boundary conditions
    K = K_ss
    R1 = (ϵ(x) + μ(x))*K[1](x,x) ~ -C[2,1](x)
    R2 = μ(0)*K[2](x,0) ~ q*ϵ(0)*K[1](x,0)
    @test K[1](x,x).series.get_selfseries_coefficients(2) == Set([K[1][2,0], K[1][1,1], K[1][0,2]])
    @test get_involved_selfseries_coefficients(R1, 2) == Set([K[1][2,0], K[1][1,1], K[1][0,2], K[1][1,0], K[1][0,1], K[1][0,0]])
    @test get_involved_selfseries_coefficients(R2, 2) == Set([K[1][2,0], K[2][2,0], K[1][1,0], K[2][1,0], K[1][0,0], K[2][0,0]])

    # main PDE
    K = K_ss(x,ξ)
    R3 = μ(x)*∂x(K[2]) + μ(ξ)*∂ξ(K[2]) ~ -∂ξ(μ(ξ))*K[2] + (C[1,2](ξ)-C[2,2](ξ))*K[1] + C[2,2](x)*K[1]
    R4 = μ(x)*∂x(K[1]) - ϵ(ξ)*∂ξ(K[1]) ~ ∂ξ(ϵ(ξ))*K[1] + (C[2,1](ξ)-C[1,1](ξ))*K[2] + C[2,2](x)*K[2]
    @test get_involved_selfseries_coefficients(R3, 0, 0) == Set([K_symbols[2][0,0], K_symbols[2][1,1], K_symbols[1][0,0], K_symbols[2][1,0]])
    @test get_involved_selfseries_coefficients(R4, 0, 0) == Set([K_symbols[1][1,0], K_symbols[1][1,1], K_symbols[1][0,0], K_symbols[2][0,0]])

    # PDESeries
    getindices(N) = N ≥ 1 ? generate_fullsym_indices(N-1, 2) : []
    K = PDESeries{Float64}(:K, [x,ξ], [0,0], K_symbols, [R1, R2, R3, R4])
    compute_coefficients!(K, N; verbose=2)

    Kᵛᵘ = K_symbols[1]
    Kᵛᵛ = K_symbols[2]
    Kᵛᵘ_matematica = [-1.11111, -0.109739, 0.109739, 0.737015/2, 1.89131/2, 1.15768/2, -1.18738/6, -5.98654/6, -6.99183/6, 16.6349/6]
    Kᵛᵛ_matematica = [-0.888889, -0.0877915, 0.0877915, 0.589612/2, -1.17922/2, 0.293316/2, -0.949904/6,  3.6947/6, -3.7825/6, 1.0377/6]
    tol = 1e-5
    for i in 0:3, j in 0:i
        @test ≈(getNum(Kᵛᵘ[i,j]), Kᵛᵘ_matematica[convertIndices_trunc_to_lin(i,j)], atol=tol)
        @test ≈(getNum(Kᵛᵛ[i,j]), Kᵛᵛ_matematica[convertIndices_trunc_to_lin(i,j)], atol=tol)
    end
end

let 

    K_symbols = selfseries_symbols(2)
    K_ss = SymbolicSeries(K_symbols, [0,0])
    
    # boundary conditions
    K = K_ss
    R1 = (ϵ(x) + μ(x))*K[1](x,x) ~ -C[2,1](x)
    R2 = μ(0)*K[2](x,0) ~ q*ϵ(0)*K[1](x,0)
    @test K[1](x,x).series.get_selfseries_coefficients(2) == Set([K[1][2,0], K[1][1,1], K[1][0,2]])
    @test get_involved_selfseries_coefficients(R1, 2) == Set([K[1][2,0], K[1][1,1], K[1][0,2], K[1][1,0], K[1][0,1], K[1][0,0]])
    @test get_involved_selfseries_coefficients(R2, 2) == Set([K[1][2,0], K[2][2,0], K[1][1,0], K[2][1,0], K[1][0,0], K[2][0,0]])
    
    # main PDE
    K = K_ss(x,ξ)
    R3 = μ(x)*∂x(K[2]) + μ(ξ)*∂ξ(K[2]) ~ -∂ξ(μ(ξ))*K[2] + (C[1,2](ξ)-C[2,2](ξ))*K[1] + C[2,2](x)*K[1]
    R4 = μ(x)*∂x(K[1]) - ϵ(ξ)*∂ξ(K[1]) ~ ∂ξ(ϵ(ξ))*K[1] + (C[2,1](ξ)-C[1,1](ξ))*K[2] + C[2,2](x)*K[2]
    @test get_involved_selfseries_coefficients(R3, 0, 0) == Set([K_symbols[2][0,0], K_symbols[2][1,1], K_symbols[1][0,0], K_symbols[2][1,0]])
    @test get_involved_selfseries_coefficients(R4, 0, 0) == Set([K_symbols[1][1,0], K_symbols[1][1,1], K_symbols[1][0,0], K_symbols[2][0,0]])

    #LocalizedPDESeries
    K = LocalizedPDESeries{Float64}(:K, [x,ξ], [0,0], [R1, R2, R3, R4], K_symbols)
    compute_coefficients!(K, N; verbose=true)

    Kᵛᵘ = K[1]
    Kᵛᵛ = K[2]
    Kᵛᵘ_matematica = [-1.11111, -0.109739, 0.109739, 0.737015/2, 1.89131/2, 1.15768/2, -1.18738/6, -5.98654/6, -6.99183/6, 16.6349/6]
    Kᵛᵛ_matematica = [-0.888889, -0.0877915, 0.0877915, 0.589612/2, -1.17922/2, 0.293316/2, -0.949904/6,  3.6947/6, -3.7825/6, 1.0377/6]
    tol = 1e-5
    for i in 0:3, j in 0:i
        @test ≈(getNum(Kᵛᵘ[i,j]), Kᵛᵘ_matematica[convertIndices_trunc_to_lin(i,j)], atol=tol)
        @test ≈(getNum(Kᵛᵛ[i,j]), Kᵛᵛ_matematica[convertIndices_trunc_to_lin(i,j)], atol=tol)
    end
end