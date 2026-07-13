using Test
using BenchmarkTools
include("../utils.jl")


@test convertIndices_fullsym_to_trunc(0,0,0) ≈ [0,0,0]
@test convertIndices_fullsym_to_trunc(1,0,0) ≈ [1,0,0]
@test convertIndices_fullsym_to_trunc(0,1,0) ≈ [1,1,0]
@test convertIndices_fullsym_to_trunc(0,0,1) ≈ [1,1,1]
@test convertIndices_fullsym_to_trunc(2,0,0) ≈ [2,0,0]
@test convertIndices_fullsym_to_trunc(1,1,0) ≈ [2,1,0]
@test convertIndices_fullsym_to_trunc(1,0,1) ≈ [2,1,1]

@test convertIndices_trunc_to_fullsym(0,0,0) ≈ [0,0,0]
@test convertIndices_trunc_to_fullsym(1,0,0) ≈ [1,0,0]
@test convertIndices_trunc_to_fullsym(1,1,0) ≈ [0,1,0]
@test convertIndices_trunc_to_fullsym(1,1,1) ≈ [0,0,1]
@test convertIndices_trunc_to_fullsym(2,0,0) ≈ [2,0,0]
@test convertIndices_trunc_to_fullsym(2,1,0) ≈ [1,1,0]
@test convertIndices_trunc_to_fullsym(2,1,1) ≈ [1,0,1]

@test generate_trunc_indices(2, 3) == [[2,0,0], [2,1,0], [2,1,1], [2,2,0], [2,2,1], [2,2,2]]

@test apply_with_fullsym_indices_from_and_upto(idcs::Vararg -> +(idcs...), +, 2, 0, 0, 0) == 15

for n in 0:10
    for k in 4:2:min(n, 8)
        @test dynamic_binomial(n, k) == binomial(n,k)
    end
end

# print("native binomial : ")
# @btime begin
    
#     for _ in 0:50
#         for n in 0:50
#             for k in 0:n
#                 binomial(Int128(n),k)
#             end
#         end
#     end

# end

# dyn_binoms_test = Dict{Tuple{Int, Int}, Int128}()
# function dynamic_binomial_test(n::Int, k::Int)::Int128
#     if k > n
#         throw(ArgumentError("$k > $n. Cannot compute binomial coefficient"))
#     elseif k > n÷2
#         return dynamic_binomial_test(n, n-k)
#     elseif (n,k) in keys(dyn_binoms_test)
#         return dyn_binoms_test[n,k]
#     elseif k == 0
#         dyn_binoms_test[n,k] = 1
#         return 1
#     else
#         dyn_binoms_test[n,k] = (n÷k)*dynamic_binomial_test(n-1, k-1)
#         return dyn_binoms_test[n,k]
#     end
# end

# print("dynamic binomial : ")
# @btime begin

#     global dyn_binoms_test = Dict{Tuple{Int, Int}, Int128}()

#     for _ in 0:50
#         for n in 0:50
#             for k in 0:n
#                 dynamic_binomial_test(n, k)
#             end
#         end
#     end

# end

#=For large binomial coefficients (n > 50), dynamic implementation is faster because one 
  has to use Int128 to store the result but when using Int64, native binomial is faster
=#

@test convertIndices_fullsym_to_lin(3,2,3)   == 139
@test convertIndices_fullsym_to_lin(1,2,2)   == 48
@test convertIndices_fullsym_to_lin(3,0,3)   == 66
@test convertIndices_fullsym_to_lin(3,4)     == 33
@test convertIndices_fullsym_to_lin(8)       == 9
@test convertIndices_fullsym_to_lin(9,4,5,2) == 9172