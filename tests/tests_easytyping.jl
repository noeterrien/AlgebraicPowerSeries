using Test

include("../AlgebraicPowerSeries.jl")

@variables x y z
@variables i j k 

#---------------------------------using 1-D reaction diffusion equation with space-varying reaction example--------------------------------

#parameters
N = 10

# (λ+c)/ε function
λ = TaylorExpansionSeries{Float64}(:sin, [x], [sin(x)], [0])
compute_coefficients!(λ, N)

# kernel representation
K = selfseries_symbols(1)

# recurrent relations
R1 = @recurrent_relation K[1][i,0] ~ 0 i in 0:(:∞)
R2 = @recurrent_relation (@∑ K[1][i,j] j in 0:i) ~ -λ[1][i-1]/(2*i) i in 1:(:∞)
R3 = @recurrent_relation (i-j)*(i-j-1)*K[1][i,j] - (j+2)*(j+1)*K[1][i,j+2] ~ (@∑ K[1][k,j]*λ[1][i-2-k] k in j:(i-2)) j in 0:(i-2) i in 2:(:∞)

# recurrent series
rs = RecurrentSeries{Float64}(:K, (1,), [x,y], [0,0], [R1,R2,R3])
compute_coefficients!(rs, N)

# tests
@test rs.coefficients[1][1:10] ≈ [0,0,0,0,-1/4,0,0,0,0,0]

#-----------------------------------------------using 2x2 1-D linear hyperbolic system------------------------------------------------------

#parameters
N=10
q=1

# Σ function
Σ = TaylorExpansionSeries{Float64}(:Σ, [x], [1+x^2;0;;0;exp(x)], [0])
compute_coefficients!(Σ, N+1); println("coefficients computed for Σ up to order $(N+1)")

# C function
C = TaylorExpansionSeries{Float64}(:C, [x], [0;cos(x);;sin(x);0], [0])
compute_coefficients!(C, N); println("coefficients computed for C up to order $N")

# kernel representation
K = selfseries_symbols(2,2)

# relations of recurrence
R11 = @recurrent_relation K[1,1][i,0] ~ Σ[2,2][0]/(q*Σ[1,1][0])*K[1,2][i,0] i in 0:(:∞)
R12 = @recurrent_relation K[2,1][i,0] ~ Σ[2,2][0]/(q*Σ[1,1][0])*K[2,2][i,0] i in 0:(:∞)

R21 = @recurrent_relation (@∑ K[1,2][k,j]*(Σ[1,1][i-k]+Σ[2,2][i-k]) j in 0:k k in 0:i) ~  C[1,2][i] i in 0:(:∞)
R22 = @recurrent_relation (@∑ K[2,1][k,j]*(Σ[1,1][i-k]+Σ[2,2][i-k]) j in 0:k k in 0:i) ~ -C[2,1][i] i in 0:(:∞)

R31 = @recurrent_relation (@∑ (k+1)*K[1,1][k+j+1,j]*Σ[1,1][i-1-j-k] k in 0:(i-1-j)) + (@∑ (k+1)*K[1,1][i-1-j+k+1,k+1]*Σ[1,1][j-k] k in 0:j) ~ -(@∑ (j-k+1)*K[1,1][i-1-j+k,k]*Σ[1,1][j-k+1] k in 0:j) - (@∑ K[1,2][i-1-j+k,k]*C[2,1][j-k] k in 0:j) j in 0:(i-1) i in 0:(:∞)
R32 = @recurrent_relation (@∑ (k+1)*K[1,2][k+j+1,j]*Σ[1,1][i-1-j-k] k in 0:(i-1-j)) - (@∑ (k+1)*K[1,2][i-1-j+k+1,k+1]*Σ[2,2][j-k] k in 0:j) ~  (@∑ (j-k+1)*K[1,2][i-1-j+k,k]*Σ[2,2][j-k+1] k in 0:j) - (@∑ K[1,1][i-1-j+k,k]*C[1,2][j-k] k in 0:j) j in 0:(i-1) i in 0:(:∞)
R33 = @recurrent_relation (@∑ (k+1)*K[2,1][k+j+1,j]*Σ[2,2][i-1-j-k] k in 0:(i-1-j)) - (@∑ (k+1)*K[2,1][i-1-j+k+1,k+1]*Σ[1,1][j-k] k in 0:j) ~  (@∑ (j-k+1)*K[2,1][i-1-j+k,k]*Σ[1,1][j-k+1] k in 0:j) + (@∑ K[2,2][i-1-j+k,k]*C[2,1][j-k] k in 0:j) j in 0:(i-1) i in 0:(:∞)
R34 = @recurrent_relation (@∑ (k+1)*K[2,2][k+j+1,j]*Σ[2,2][i-1-j-k] k in 0:(i-1-j)) + (@∑ (k+1)*K[2,2][i-1-j+k+1,k+1]*Σ[2,2][j-k] k in 0:j) ~ -(@∑ (j-k+1)*K[2,2][i-1-j+k,k]*Σ[2,2][j-k+1] k in 0:j) + (@∑ K[2,1][i-1-j+k,k]*C[1,2][j-k] k in 0:j) j in 0:(i-1) i in 0:(:∞)

# recurrent series
coupled_rs = RecurrentSeries{Float64}(:crs, (2,2), [x,y], [0,0], [R11, R12, R21, R22, R31, R32, R33, R34])
compute_coefficients!(coupled_rs, N; verbose=1)

# tests
@test coupled_rs.coefficients[1,1][1:6] ≈ [0, 1/4, -1/4, 3/32, -7/16, 3/32]
@test coupled_rs.coefficients[1,2][1:6] ≈ [0, 1/4, 1/4, 3/32, -1/16, -9/32]
@test coupled_rs.coefficients[2,1][1:6] ≈ [-1/2, -1/8, 3/8, 5/64, 5/32, 17/64]
@test coupled_rs.coefficients[2,2][1:6] ≈ [-1/2, -1/8, 5/8, 5/64, 3/32, -43/64]

built_function = build(coupled_rs, 2)
@test built_function(0,0) ≈ [0;-1/2;;0;-1/2]
@test built_function(1,1) ≈ [-1/4;1/4;;1/4;-1/2]
@test built_function(1,2) ≈ [-21/32;101/64;;-13/32;-115/64]