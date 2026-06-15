using Symbolics

"""
    function substitute_in_pairs_vector(lst::Vector{Tuple{Any,Any}}, d::Dict)

    Applies the Symbolics.substitute function to all elements of a vector of pairs
    and returns the result

    ###Input

    - `lst::Vector` -- Apply substitute to all elements of the pairs of this vector
    - `d::Dict` -- Apply substitute with this dictionnary of associations

    ###Output

    A Vector{Tuple{Any, Any}} containing the substituted expressions
"""
function substitute_in_pairs_vector(lst::Vector, d::Dict)
    res = []
    for t in lst
        push!(res, (
            substitute(t[1], d),
            substitute(t[2], d)
        ))
    end
    res
end

"""
    function substitute_in_vector(lst::Vector, d::Dict)

    Applies the Symbolics.substitute function to all elements of the vector
    and returns the result

    ###Input

    - `lst::Vector` -- Apply substitute to all elements of this vector
    - `d::Dict` -- Apply substitute with this dictionnary of associations

    ###Output

    A Vector containing the substituted expressions
"""
substitute_in_vector(lst::Vector, d::Dict) = [substitute(x, d) for x in lst]

intparse(s::String) = parse(Int, s)



"""
    decode_IntVector(t::String)::Vector{Int}

    Parses a Vector of Int written as a string to a Vector

    ###Input

    -  `t::String` -- a String of the form "a, b, c, ..." a, b, c, ... isa Int
    
    ###Output

    The Vector [a, b, c, ...]
"""
function decode_IntVector(t::String)::Vector{Int}
    str_res = [""]
    for c in t
        if c == ','
            push!(str_res, "")
        elseif c != ' ' # c is a number, and skip spaces
            str_res[end] *= c
        end
    end

    intparse.(str_res)
end

"""
    convertIndices_trunc_to_lin(I::Vararg{Int64})

    Converts indices of the form a₀₀₀ + a₁₀₀ x + a₁₁₀ y + a₁₁₁ z + a₂₀₀ x² + a₂₁₀ xy + 
    a₂₁₁ xz + a₂₂₀ y² + a₂₂₁ yz + a₂₂₂ z² + ... to an index of the the form 
    b₁ + b₂ x + b₃ y + b₄ z + b₅ x² + b₆ xy + b₇ xz + b₈ y² + b₉ yz + b₁₀ z² + ... 
"""
function convertIndices_trunc_to_lin(I::Vararg{Int64})
    fst_nonzero = isnothing(findfirst(==(0), I)) ? length(I) : findfirst(==(0), I) - 1
    if fst_nonzero == 0
        return 1
    else
        new_idx = [I[1:fst_nonzero-1]..., [I[fst_nonzero]-1 for _ in fst_nonzero:length(I)]...]
        return 1 + convertIndices_trunc_to_lin(new_idx...)
    end
end

"""
    convertIndices_fullsym_to_trunc(I:vararg{Int64})

    Converts indices of the form a₀₀₀ + a₁₀₀ x + a₀₁₀ y + a₀₀₁ z + a₁₁₀ xy + ...
    to indices of the form a₀₀₀ + a₁₀₀ x + a₁₁₀ y + a₁₁₁ z + a₂₀₀ x² + a₂₁₀ xy + 
    a₂₁₁ xz + a₂₂₀ y² + a₂₂₁ yz + a₂₂₂ z² + ...

    ###Input

    - `I::Vararg{Int64}` -- The input index

    ###Output

    A vector of the output index
"""
convertIndices_fullsym_to_trunc(I::Vararg{Int64}) = [+(I[i:end]...) for i in eachindex(I)]

convertIndices_trunc_to_fullsym(I::Vararg{Int64}) = begin J = [I..., 0] ; [J[i]-J[i+1] for i in eachindex(I)] end

"""
    decode_coeffIndex(u_sym::Num)::Tuple{Vector{Int}, Int}

    Returns a Tuple{Vector{Int}, Int}, the vector representing the coefficient index in the
    matrix of series and the second element representing the index of the coefficient in
    the series coefficients vector

    ###Input
    
    - `u_sym::Num` -- The unique symbol representing the coefficient in an expression.
      This decoding function should work with any unique representation that have the
      form "...(a, b, c, ...)...[d, e, f, ...]" where a, b, c,... are the series index
      in the matrix of series and d, e, f, ... are the coefficient indices (The number of
      indices is equal to the number of variables)

    ###Output

    A Tuple{Vector{Int}, Int}
"""
function decode_coeffIndexAndIndices(u_sym::Num)::Tuple{Vector{Int}, Int}

    str_sym = string(u_sym)
    
    # matrix index
    beg_parenthesis = findfirst('(', str_sym)
    end_parenthesis = findfirst(')', str_sym)

    str_sym[end_parenthesis-1] == ',' && (end_parenthesis -= 1) # handle 1D case
    
    midx = decode_IntVector(str_sym[beg_parenthesis+1:end_parenthesis-1])


    # series coefficient index
    beg_parenthesis = findfirst('[', str_sym)
    end_parenthesis = findfirst(']', str_sym)

    sidc = decode_IntVector(str_sym[beg_parenthesis+1:end_parenthesis-1])

    # return
    midx, convertIndices(sidc...)

end


"""
    compute_monomials(N::Int, variables::Vector{Num}, center::Vector)::Vector{Num}

    Computes the monomials of variables up to order N

    ###Input

    - `N::Int` -- The order up to which the monomials should be computed
    - `variables::Vector{Num}` -- The variables these monomials migth depend on
      (assume they commute !)
    - `center::Vector` -- The center around which to compute the monomials. Muste be the
      same length as variables
      
    ###Output

    A Vector{Num} of the monomials in increasing order. For instance, for 3 variables
    x, y, z, it would be [Num(1), x, y, z, x², xy, xz, y², yz, z², ...]
"""
function compute_monomials(N::Int, variables::Vector{Num}, center::Vector)::Vector{Num}
    monomials = Num[Num(1)]
    mult_start_at = ones(Int, length(variables))

    for _ in 1:N
        l_monom = length(monomials)
        for ((i,v),c) in zip(enumerate(variables), center)
            for j in mult_start_at[i]:l_monom
                push!(monomials, (v-c)*monomials[j])
            end
            mult_start_at[i] += length(monomials)-l_monom
        end
    end
    
    monomials
end

"""
    getAllVariables(exprs::Vector)::Vector{Num}

    Returns a Vector{Num} of all the variables that appear in a vector of expressions

    ###Input

    - `exprs::Vector` -- A Vector of Symbolics expressions

    ###Output

    The Vector of all the variables that appear in exprs
"""
function getAllVariables(exprs::Vector)::Vector{Num}
    res = Set{Num}()
    for expr in exprs
        res = union(res, Symbolics.get_variables(expr))
    end
    Vector([res...])
end

generate_index_list(D::Int) = [Symbol("idx$i") for i in 1:D]

"""
    generate_trunc_indices(N::Int, D::Int)

    Generates the truncated indices of order N and dimension D
    For instance, 
    generate_trunc_indices(2, 3) = [[2,0,0], [2,1,0], [2,1,1], [2,2,0], [2,2,1], [2,2,2]]
"""
function generate_trunc_indices(N::Int, D::Int)::Vector
    if D==1
        [N]
    else
        prev_res = generate_trunc_indices(N, D-1)
        res = []
        for idx in prev_res
            for i in 0:idx[end]
                push!(res, [idx..., i])
            end
        end
        res
    end
end

"""
    generate_fullsym_indices(N::Int, D::Int)

    Generates the fully symetric indices of order N and dimension D
    For instance, 
    generate_fullsym_indices(2, 3) = [[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]]
"""
generate_fullsym_indices(N::Int, D::Int)::Vector = map(idx -> convertIndices_trunc_to_fullsym(idx...), generate_trunc_indices(N,D))