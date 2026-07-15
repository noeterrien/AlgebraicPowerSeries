######################################## Imports ##########################################
import LinearSolve as LS
using MUMPS
using SparseArrays
using CUDSS
using CUDA
import ParU_jll

################################# Dense Solvers ###################################

function julia_default(lp::LS.LinearProblem; benchmark=false) 
    t1 = time()
    res = lp.A \ lp.b
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function QRFactorization(lp::LS.LinearProblem; benchmark = false)
    t1 = time()
    res = LS.solve(lp, LS.QRFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function CudaOffloadLUFactorization(lp::LS.LinearProblem; benchmark = false)
    t1 = time()
    res = LS.solve(lp, LS.CudaOffloadLUFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function CudaOffloadQRFactorization(lp::LS.LinearProblem; benchmark = false)
    t1 = time()
    res = LS.solve(lp, LS.CudaOffloadQRFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

###################################### Sparse Solvers #####################################

function sparse_SparseColumnPivotedQRFactorization(lp::LS.LinearProblem; benchmark=false)
    sparse_lp = LS.LinearProblem(sparse(lp.A), lp.b)
    t1 = time()
    res = LS.solve(sparse_lp, LS.SparseColumnPivotedQRFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function sparse_QRFactorization(lp::LS.LinearProblem; benchmark=false)
    sparse_lp = LS.LinearProblem(sparse(lp.A), lp.b)
    t1 = time()
    res = LS.solve(sparse_lp, LS.QRFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function sparse_ParUFactorization(lp::LS.LinearProblem; benchmark=false)
    sparse_lp = LS.LinearProblem(sparse(lp.A), lp.b)
    t1 = time()
    res = LS.solve(sparse_lp, LS.ParUFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end

function sparse_MUMPSFactorization(lp::LS.LinearProblem; benchmark=false)
    sparse_lp = LS.LinearProblem(sparse(lp.A), lp.b)
    t1 = time()
    res = LS.solve(sparse_lp, LS.MUMPSFactorization())
    t2 = time()
    benchmark && println("solving took $(t2-t1) seconds")
    res
end