######################################## Imports ##########################################
using LinearSolve
using MUMPS

using SparseArrays
using CUDSS
################################# Miscellaneous Solvers ###################################

julia_default(lp::LinearProblem) = lp.A \ lp.b

###################################### Sparse Solvers #####################################

function sparse_PureKLUFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    PureKLUFactorization(sparse_lp)
end

function sparse_UMFPACKFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    UMFPACKFactorization(sparse_lp)
end

function sparse_SparseColumnPivotedQRFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    PureKLUFactorization(sparse_lp)
end

function sparse_QRFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    QRFactorization(sparse_lp)
end

function sparse_ParUFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    ParUFactorization(sparse_lp)
end

function sparse_MUMPSFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    MUMPSFactorization(sparse_lp)
end

function sparse_CUDALUFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(CudaSparseMatrixCSR(lp.A), lp.b)
    LUFactorization(sparse_lp)
end

function sparse_CUSOLVERRFFactorization(lp::LinearProblem)
    sparse_lp = LinearProblem(CudaSparseMatrixCSR(lp.A), lp.b)
    CUSOLVERRFFactorization(sparse_lp)
end

function sparse_CPUKrylovJL_CG(lp::LinearProblem)
    sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    KrylovJL_CG(sparse_lp)
end

function sparse_CPUKrylovJL_GMRES(lp::LinearProblem)
sparse_lp = LinearProblem(sparse(lp.A), lp.b)
    KrylovJL_CPUKrylovJL_GMRES(sparse_lp)
end

function sparse_CUDAKrylovJL_CG(lp::LinearProblem)
    sparse_lp = LinearProblem(CudaSparseMatrixCSR(lp.A), lp.b)
    KrylovJL_CG(sparse_lp)
end

function sparse_CUDAKrylovJL_GMRES(lp::LinearProblem)
sparse_lp = LinearProblem(CudaSparseMatrixCSR(lp.A), lp.b)
    KrylovJL_CPUKrylovJL_GMRES(sparse_lp)
end