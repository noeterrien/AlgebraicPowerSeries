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
    get_matrix_index(c_lab::String)

    Retrieves the matrix index of a series coefficient from its unique identifier

    ###Input
    - `c_lab::String` -- A series coefficient unique identifier, parsed as a String using
      string(::Num)

    ###Output
    A Vector{Int64} representing the index of the coefficient in the matrix
"""
function get_matrix_index(c_lab::String)
    beg_parenthesis = findfirst('(', c_lab)
    end_parenthesis = findfirst(')', c_lab)

    str_idx = [""]
    for c in c_lab[beg_parenthesis+1:end_parenthesis-1]
        if c == ','
            push!(str_idx, "")
        elseif c != ' ' # c is a number, skip spaces
            str_idx[end] *= c
        end
    end

    str_idx = str_idx[end] == "" ? str_idx[1:end-1] : str_idx # Handle 1-dimensional case

    intparse.(str_idx)

end


"""
    order_lex(labels::Vector, values::Vector)

    Orders labels in lexicographical order and performs the corresponding reordering in
    values Vector, returning the vector of values

    ###Input

"""
order_lex(labels::Vector{String}, values::Vector) = values[sortperm(labels)]