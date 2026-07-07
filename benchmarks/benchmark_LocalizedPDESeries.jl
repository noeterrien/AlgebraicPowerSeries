######################################### Imports #########################################

using BenchmarkTools
import LinearSolve as LS

include("../AlgebraicPowerSeries.jl")
include("../custom_solvers.jl")

################################## benchmark Parameters ###################################

# Where the results should be saved (if empty, prints to the console)
save_to = "benchmark_LocalizedPDESeries.txt"

iszero_tol = 1e-15

maxOrder = 10

# The solvers to test
solvers = [LS.QRFactorization(), LS.QRFactorization(), 
           julia_default, julia_default,
           LS.RFLUFactorization, LS.RFLUFactorization,
           LS.OpenBLASLUFactorization, LS.OpenBLASLUFactorization,
           LS.LUFactorization, LS.LUFactorization,
           LS.CudaOffloadLUFactorization, LS.CudaOffloadLUFactorization,
           LS.CudaOffloadQRFactorization, LS.CudaOffloadQRFactorization,
           CUDAOffload32MixedLUFactorization,
           sparse_PureKLUFactorization,
           sparse_UMFPACKFactorization,
           sparse_SparseColumnPivotedQRFactorization,
           sparse_QRFactorization,
           sparse_ParUFactorization,
           sparse_MUMPSFactorization,
           sparse_CUDALUFactorization,
           sparse_CUSOLVERRFFactorization,
           sparse_CPUKrylovJL_CG,
           sparse_CPUKrylovJL_GMRES,
           sparse_CUDAKrylovJL_CG, sparse_CUDAKrylovJL_CG,
           sparse_CUDAKrylovJL_GMRES, sparse_CUDAKrylovJL_GMRES]

# The name of each solver
solver_names = ["QRFactorization", "QRFactorization", 
                "julia default", "julia default",
                "RFLUFactorization", "RFLUFactorization",
                "OpenBLASLUFactorization", "OpenBLASLUFactorization",
                "LUFactorization", "LUFactorization",
                "CudaOffloadLUFactorization", "CudaOffloadLUFactorization",
                "CudaOffloadQRFactorization", "CudaOffloadQRFactorization",
                "CUDAOffload32MixedLUFactorization",
                "sparse PureKLUFactorization",
                "sparse UMFPACKFactorization",
                "sparse SparseColumnPivotedQRFactorization",
                "sparse QRFactorization",
                "sparse ParUFactorization",
                "sparse MUMPSFactorization",
                "sparse CUDALUFactorization",
                "sparse CUSOLVERRFFactorization",
                "sparse CPUKrylovJL_CG",
                "sparse CPUKrylovJL_GMRES",
                "sparse CUDAKrylovJL_CG", "sparse CUDAKrylovJL_CG",
                "sparse CUDAKrylovJL_GMRES", "sparse CUDAKrylovJL_GMRES"]

# The type associated with each solver
types = [Float64, Float32, 
         Float64, Float32,
         Float64, Float32,
         Float64, Float32,
         Float64, Float32,
         Float64, Float32,
         Float64, Float32,
         Float32,
         Float64,
         Float64,
         Float64,
         Float64,
         Float64,
         Float64,
         Float32,
         Float64,
         Float64,
         Float64,
         Float64, Float32,
         Float64, Float32]

####################################### The problem #######################################

@variables x y
∂²x, ∂²y = Differential(x)^2, Differential(y)^2

center = [0.5, 0.7]

c = 3
ε = 1

"""
    benchmark_with(T::TypeVar, solver)

    benchmarks how a fast a solver is

    ### Input

    - `T::TypeVar` -- The expected type of the coefficients
    - `solver` -- The solver to be used. The solver should take as input a 
      LinearSolve.LinearProblem and return a Vector of the coefficients values. These 
      values should be castable to type T.
    - `solver_name::String` -- The name of the solver, will be used to display the results
    - `save_to=nothing` -- If nothing, the results of the benchmark are printed in the
      console. Otherwise, one can pass a file name and the results will be written at 
      benchmarks/results/save_to instead

    ### Output

    nothing

"""
function benchmark_with(T::Type, solver, solver_name::String)

    ####### replace this part with your own problem #######

    λ_ps = TaylorExpansionSeries{T}(:λ, [x], [√(0.5 + x^2)], [center[2]])
    compute_coefficients!(λ_ps, maxOrder)
    λ = SymbolicSeries(λ_ps)

    unknown = selfseries_symbols()
	K = SymbolicSeries(unknown, center)

    BC1 = K(x,0) ~ 0
	BC2 = K(x,x) ~ -1/(2*ε) * ∫(λ(x) + c, 0, x, x)
	PDE = ∂²x(K(x,y)) - ∂²y(K(x,y)) ~ K(x,y) * (λ(y)+c)/ε

    K_ps = LocalizedPDESeries{T}(:K, [x,y], center, [BC1, BC2, PDE], unknown)
    
    #######################################################

    print("Benchmark results for solver $solver_name with type $T: ")
    @btime compute_coefficients!($K_ps, $maxOrder; solver=$solver)
    println()

end

################################### Benchmark all solvers #################################

redirect_stdio(stdout=isempty(save_to) ? stderr : "benchmarks/results/$save_to", 
               stderr=devnull) do

    for (solver, solver_name, T) in zip(solvers, solver_names, types)
        benchmark_with(T, solver, solver_name)
    end

end