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