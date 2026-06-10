using Test
include("../utilitaries.jl")


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