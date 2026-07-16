######################################### Imports #########################################

include("../AlgebraicPowerSeries.jl")

################################## benchmark Parameters ###################################

# Where the results should be saved (if empty, prints to the console)
save_to = "quickbenchmark_LocalizedPDESeries.txt"

iszero_tol = 1e-15

compileOrder = 1 # used to compile the required functions before measuring the time needed
maxOrder = 50

# The solvers to test
solvers = [QRFactorization, QRFactorization, 
           julia_default, julia_default,
           CudaOffloadQRFactorization, CudaOffloadQRFactorization,
           sparse_SparseColumnPivotedQRFactorization,
           sparse_QRFactorization,
           sparse_ParUFactorization,
           sparse_MUMPSFactorization]

# The name of each solver
solver_names = ["QRFactorization", "QRFactorization", 
                "julia default", "julia default",
                "CudaOffloadQRFactorization", "CudaOffloadQRFactorization",
                "sparse SparseColumnPivotedQRFactorization",
                "sparse QRFactorization",
                "sparse ParUFactorization",
                "sparse MUMPSFactorization",]

# The type associated with each solver
types = [Float64, Float32, 
         Float64, Float32,
         Float64, Float32,
         Float64,
         Float64,
         Float64,
         Float64,]

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


    # compile everything
    compute_coefficients!(K_ps, compileOrder; solver=solver)

    println("Computing coefficients with solver $solver_name and type $T:")

    @time compute_coefficients!(K_ps, maxOrder; solver=solver, benchmark=true)

    println()

end

################################### Benchmark all solvers #################################

redirect_stdio(stdout=isempty(save_to) ? stdout : "benchmarks/results/$save_to", 
               stderr=devnull) do

    for (solver, solver_name, T) in zip(solvers, solver_names, types)
        benchmark_with(T, solver, solver_name)
    end

end