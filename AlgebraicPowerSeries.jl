using Symbolics
import TaylorSeries
include("utils.jl")
include("solvers.jl")

#---------------------------------------------------------------------------------------------------------------------
abstract type AbstractPowerSeries{D} end
abstract type AbstractScalarSeriesSymbol end

"""
    PowerSeries{T,D}

    An abstract type to represent algebraic multivariate, multidimensional power series.
     
    T is the type of the coefficients
    D is the number of dimensions (0 = scalar series, 1 = vector series, ...)

    ### Notes

    Every concrete PowerSeries{T,D} must have the following fields and methods :
    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients. Multivariate series
      are represented as a₀₀₀ + a₁₀₀ x + a₁₁₀ y + a₁₁₁ z + a₂₀₀ x² + a₂₁₀ xy + a₂₁₁ xz +
      a₂₂₀ y² + a₂₂₁ yz + a₂₂₂ z² + ... thus the coefficients vectors are 
      [a₀₀₀, a₁₀₀, a₁₁₀, a₁₁₁, a₂₀₀, a₂₁₀, a₂₁₁, a₂₂₀, a₂₂₁, a₂₂₂, ...]
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `scalar_series_ref::Array{AbstractScalarSeriesSymbol, D}` -- An array of 
      AbstractScalarSeriesSymbol that is used to easily create and access SeriesCoefficient

    - `compute_coefficients!(ps::PowerSeries, N::Int)` -- computes the coefficients up to 
      order N
"""
abstract type PowerSeries{T,D} <: AbstractPowerSeries{D} end  

function Base.show(io::IO, ps::PowerSeries)

    if ps.order >= 0
        monomials = compute_monomials(ps.order, ps.variables, ps.center)
        to_build = zeros(Num, ps.size)
        for i in eachindex(ps.coefficients)
            to_build[i] = sum(ps.coefficients[i][1:length(monomials)] .* monomials)
        end
        print(io, "PowerSeries ", ps.seriesID, " with coefficients : ", to_build)
    else 
        print(io, "PowerSeries ", ps.seriesID, ", no coefficients computed yet")
    end
end

"""
    build_matrix_elt(ps::PowerSeries, N::Int)

    Builds the polynomial function from the series coefficients. Returns a matrix of
    functions instead of a function that returns matrices

    ###Input

    - `ps::PowerSeries` -- The PowerSeries from which one wants to compute the function
    - `N::Int` -- The order up to which the polynomial should be built. If not given,
      computes up to the series current order

    ###Output

    A matrix of function of ps variables. (i.e if ps is a PowerSeries of n variables, and
    size (a,b), a matrix of size (a,b) of functions of the n variables, in the same order)

    Returns the null function if N<0
"""
function build_matrix_elt(ps::PowerSeries, N::Int)

    if N > ps.order 
        compute_coefficients!(ps, N)
    end

    # handle N<0 case
    N >= 0 || return function(args...)
        length(args) == length(ps.variables) || throw(ArgumentError("Wrong number of arguments"))
        zeros(eltype(args), ps.size)
    end
    
    monomials = compute_monomials(N, ps.variables, ps.center)
    to_build = zeros(Num, ps.size)
    for i in eachindex(ps.coefficients)
        to_build[i] = sum(ps.coefficients[i][1:length(monomials)] .* monomials)
    end

    map(expr -> build_function(expr, ps.variables...; expression=Val{false}), to_build)
end

build_matrix_elt(ps::PowerSeries) = build_matrix_elt(ps, ps.order)

"""
    build(ps::PowerSeries, N::Int)

    Builds the polynomial function from the series coefficients. 

    ###Input

    - `ps::PowerSeries` -- The PowerSeries from which one wants to compute the function
    - `N::Int` -- The order up to which the polynomial should be built. If not given,
      computes up to the current series order

    ###Output

    A function of ps variables. (i.e if ps is a PowerSeries of n variables, a function of
    n variables in the same order)

    Returns the null function if N<0
"""
function build(ps::PowerSeries, N::Int)

    built = build_matrix_elt(ps, N)

    return function(args...)
        length(args) == length(ps.variables) || throw(ArgumentError("Wrong number of arguments"))
        map(b -> b(args...), built)
    end
end

build(ps::PowerSeries) = build(ps, ps.order)



#-------------------------------------------------------SeriesCoefficient-------------------------------------------------------

"""
    SeriesCoefficient{D}

    A type to represent a PowerSeries coefficient. 

    ### Fields

    - `ps::Union{AbstractPowerSeries{D}, Symbol}` -- The PowerSeries of which it is the 
      coefficient. Use :self to refer to a PowerSeries it is given to as an argument (for 
      instance a RecurrentSeries)
    - `sym::Num` -- The symbol representing the coefficient
    - `unique_sym::Num` -- A unique symbolic representation automatically attributed to 
      identify the coefficient throughout several different expressions
    - `indices::Vector{Int}` -- The index of a scalar series coefficient
    - `index::NTuple{Int, D}` -- The index of the scalar series it refers to when using
      multidimensional series

    ### Examples

    - `SeriesCoefficient(ps::Union{AbstractPowerSeries{D}, Symbol}, 
                         sym::Num,
                         indices::Vector{Int},
                         index::NTuple{Int, D}) where D` -- default constructor

"""
mutable struct SeriesCoefficient{D}
    ps::Union{AbstractPowerSeries{D}, Symbol}
    sym::Num
    unique_sym::Num
    indices::Vector{Int}
    index::NTuple{D, Int}
end

function SeriesCoefficient(ps::Union{AbstractPowerSeries{D}, Symbol},
                           sym::Num,
                           indices::Vector{Int},
                           index::NTuple{D,Int}) where D

    if ps == :self
        u_sym = Symbol("self$index$indices") # unique id
    else
        u_sym = Symbol("$(ps.seriesID)$index$indices") # unique id
    end
    u_sym, = Symbolics.variable(u_sym)

    SeriesCoefficient(ps, sym, u_sym, indices, index)
end

Base.show(io::IO, sc::SeriesCoefficient) = print(io, sc.unique_sym)

getOrder(sc::SeriesCoefficient) = sc.indices[1]



#--------------------------------------------------------ScalarSeriesSymbol-------------------------------------------------------------

"""
    ScalarSeriesSymbol <: AbstractScalarSeriesSymbol

    A convenient way to define SeriesCoefficients

    ###Fields

    - `ps::Union{Nothing, PowerSeries, Symbol}` -- The PowerSeries it refers to or :self
    - `scalar_idx::Tuple` -- The index of the ScalarSeries in the PowerSeries matrix
    - `coefficients::Dict{Tuple, SeriesCoefficient}` -- The SeriesCoefficients stored

    ###Examples
    - `ScalarSeriesSymbol(ps::Union{PowerSeries, Symbol}, scalar_idx::Tuple, 
                          coefficients::vector{SeriesCoefficient})` -- default constructor
    - `ps[1,2]` -- (where ps is a PowerSeries) get a ScalarSeriesSymbol using the getindex
      method
"""
mutable struct ScalarSeriesSymbol <: AbstractScalarSeriesSymbol
    ps::Union{Nothing, PowerSeries, Symbol}
    scalar_idx::Tuple
    coefficients::Dict{Vector, SeriesCoefficient}
end

"""
    Base.getindex(ps::PowerSeries{D}, I::Vararg{Int64, D}) where D

    Returns the ScalarSeriesSymbol at index I

    ###Input

    - `ps::PowerSeries{D}` -- a PowerSeries
    - `I::Vararg{Int, D}` -- The index at which one wants to retrieve the 
      ScalarSeriesSymbol

    ###Output

    - The ScalarSeriesSymbol of ps at index I
"""
Base.getindex(ps::AbstractPowerSeries{D}, I::Vararg{Int64, D}) where D = ps.scalar_series_ref[I...]

"""
    Base.getindex(sss::ScalarSeriesSymbol, I::Vararg{Int})

    Create (if necessary) and returns a SeriesCoefficient at index I. The coefficient is
    then saved in the ScalarSeriesSymbol to be accessible at any time
"""
function Base.getindex(sss::ScalarSeriesSymbol, I::Vararg{Int})
    vI = [I...]
    if vI ∈ keys(sss.coefficients)
        sss.coefficients[vI]
    else
        if sss.ps == :self
            sym = Symbol("self$(sss.scalar_idx)$vI")
            sym, = Symbolics.variable(sym)
            sss.coefficients[vI] = SeriesCoefficient(:self, sym, vI, sss.scalar_idx)
        elseif sss.ps isa PowerSeries
            sym = Symbol("$(sss.ps.seriesID)$(sss.scalar_idx)$vI")
            sym, = Symbolics.variable(sym)
            sss.coefficients[vI] = SeriesCoefficient(sss.ps, sym, vI, sss.scalar_idx)
        end
        sss.coefficients[vI]
    end
end

function Base.show(io::IO, sss::ScalarSeriesSymbol)
    if sss.ps == :self
        print(io, "ScalarSeriesSymbol refering to series :self at index ", sss.scalar_idx)
    else
        if sss.ps.order < 0
            print(io, "ScalarSeriesSymbol refering to series ", sss.ps.seriesID, ". No coefficients computed yet")
        else
            monomials = compute_monomials(sss.ps.order, sss.ps.variables, sss.ps.center)
            to_build = sum(sss.ps.coefficients[sss.scalar_idx...][1:length(monomials)] .* monomials)
            print(io, "ScalarSeriesSymbol refering to series ", sss.ps.seriesID, " at index ", sss.scalar_idx,
                      ". Computed coefficients are : ", to_build)
        end
    end
end

"""
    selfseries_symbols(size::Vararg{Int})

    Generates the ScalarSeriesSymbol with reference to the series :self of size size.
    After using K = selfseries_symbols(m, n), one can then easily generate 
    SeriesCoefficients using K[scalar_series_idx][coefficient_indices]

    ###Input 

    - `size::Vararg{Int}` -- The size of the series array. If no argument is given, returns
      a ScalarSeriesSymbol

    ###Output

    An array of size size containing ScalarSeriesSymbol that refer to series :self
"""
function selfseries_symbols(size::Vararg{Int}) 
    if length(size) == 0
        ScalarSeriesSymbol(:self, (1,), Dict())
    else
        ci = reshape(collect(CartesianIndices(size)), size)
        map(idx -> ScalarSeriesSymbol(:self, Tuple(idx), Dict()), ci)
    end
end

#-----------------------------------------------------------TaylorSeries-------------------------------------------------------------


"""
    TaylorExpansionSeries{T,D} <: PowerSeries{T,D}

    A concrete type representing the Taylor development of a function around a center c

    ### Fields

    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `func::Array{Num, D}` -- Symbolics representation of a multidimensional function of the
      variables. func size is size
    - `scalar_series_ref::Array{ScalarSeriesSymbol, D}` -- An array of 
      ScalarSeriesSymbol that is used to easily create and access SeriesCoefficient

    ### Notes
    
    Tested types T include Float64 and ComplexF64. T is infered 

    ### Examples

    - `TaylorExpansionSeries{T}(seriesID::Symbol,
                                variables::Vector{Num}, 
                                func::Array{Num, D},
                                center::Vector)` -- default constructor
"""
mutable struct TaylorExpansionSeries{T,D} <: PowerSeries{T,D}
    
    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    scalar_series_ref::Array{ScalarSeriesSymbol, D}
    func::Array{Num, D}

    function TaylorExpansionSeries{T,D}(
        seriesID::Symbol,
        size::NTuple{D,Int},
        variables::Vector{Num},
        center::Vector,
        coefficients::Array{Vector{T},D},
        order::Int,
        scalar_series_ref::Array{ScalarSeriesSymbol, D},
        func::Array{Num, D}) where {T,D}

        ts = new(seriesID,
                 size,
                 variables,
                 center,
                 coefficients,
                 order,
                 scalar_series_ref,
                 func)
        for sss in scalar_series_ref
            sss.ps = ts
        end
        return ts
    end
end

function TaylorExpansionSeries{T}(seriesID::Symbol,
                        variables::Vector{Num}, 
                        func::Array{Num, D}, 
                        center::Vector) where {T,D}
    if length(variables)==length(center)
        size = Base.size(func)
        # create coefficients array
        coeffs = Array{Vector{T}}(undef, size)
        scalar_series_ref = map(idx -> ScalarSeriesSymbol(nothing, Tuple(idx), Dict()), keys(coeffs))

        TaylorExpansionSeries{T,D}(seriesID, size, variables, center, coeffs, -1, 
                              scalar_series_ref, func)
    else
        throw(ArgumentError("center size does not match number of variables"))
    end
end

"""
    compute_coefficients!(ps::TaylorSeries, N::Int)

    Computes the coefficients of a TaylorSeries up to order N

    ###Input 
    
    - `ps::TaylorExpansionSeries` -- a TaylorSeries
    - `N::Int` -- The order up to which coefficients should be computed

    ###Output

    Nothing
"""
function compute_coefficients!(ps::TaylorExpansionSeries{T}, N::Int) where T
    # first check to which order coefficients have already been computed
    ps.order >= N && return

    # Compute up to order N
    js = TaylorSeries.JetSpace(order=N, variables=Symbol.(ps.variables))
    js_var = TaylorSeries.variables(js)

    for i in eachindex(ps.coefficients)
        expr = ps.func[i]
        f = build_function(expr, ps.variables...; expression=Val{false})
        vectorized_f(v) = f.((v+ps.center)...)
        dvpt = vectorized_f(js_var)
        # ensure the returned dvpt is a TaylorSeries and not a constant
        dvpt = dvpt isa TaylorSeries.AbstractSeries ? dvpt : dvpt * one(first(js_var))

        ps.coefficients[i] = []
        for same_order_coeffs in dvpt.coeffs
            for coeff in same_order_coeffs
                push!(ps.coefficients[i], coeff)
            end
        end
    end
    
    ps.order=N
end

#-----------------------------------------------------------TranslatedSeries---------------------------------------------------------------

"""
    TranslatedSeries{T,D} <: PowerSeries{T,D}

    A concrete type representing a PowerSeries translated at a new center

    ### Fields

    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series new center
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `scalar_series_ref::Array{ScalarSeriesSymbol, D}` -- An array of 
      ScalarSeriesSymbol that is used to easily create and access SeriesCoefficient
    - `ref::PowerSeries{T,D}` -- The PowerSeries to translate

    ### Examples

    - `TranslatedSeries(seriesID::Symbol, ref::PowerSeries, new_center::Vector)` -- default 
      constructor
"""
mutable struct TranslatedSeries{T,D} <: PowerSeries{T,D}
    
    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    scalar_series_ref::Array{ScalarSeriesSymbol, D}
    ref::PowerSeries{T,D}

    function TranslatedSeries(seriesID::Symbol, ref::PowerSeries{T,D}, new_center::Vector) where {T,D}

        length(new_center) == length(ref.center) || throw(ArgumentError("New center must \
                                                    have the same size as previous center"))


        coeffs = map(x -> T[], CartesianIndices(ref.size))
        scalar_series_ref = map(idx -> ScalarSeriesSymbol(nothing, Tuple(idx), Dict()), keys(coeffs))

        ps = new{T,D}(seriesID,
                 ref.size,
                 ref.variables,
                 new_center,
                 coeffs,
                 -1,
                 scalar_series_ref,
                 ref)
        for sss in scalar_series_ref
            sss.ps = ps
        end
        return ps
    end
end


"""
    compute_coefficients!(ps::TranslatedSeries, N::Int; trunc_order=N)

    Computes the coefficients of a TranslatedSeries up to order N. Translating a series
    requires truncation. To compute the coefficients of a translated series, the reference
    series' coefficients are first computed up to order trunc_order and then used to
    compute an approximation of the TranslatedSeries coefficients, truncated at order
    trunc_order.

    ###Input 
    
    - `ps::TranslatedSeries` -- a TranslatedSeries
    - `N::Int` -- The order up to which coefficients should be computed
    - `trunc_order=N` -- The truncation order. Cannot be smaller than N

    ###Output

    The order up to which coefficients have been computed
"""
function compute_coefficients!(ps::TranslatedSeries{T}, N::Int; trunc_order=N) where T

    # first compute coefficients of the series to translate
    compute_coefficients!(ps.ref, trunc_order)

    # generate the indices of the coefficients to compute
    nbr_vars = length(ps.variables)
    indices = generate_fullsym_indices_upto(N, nbr_vars)

    # make room for the coefficients
    nbr_coeffs = length(indices)
    foreach(idx -> resize!(ps.coefficients[idx], nbr_coeffs), eachindex(ps.coefficients))

    # compute the new coefficients and fill in
    for scalar_idx in CartesianIndices(ps.size)
        for idx in indices
            if trunc_order > 66 # binomial will overflow Int64
                # construct function to compute the coefficients
                function term1(k::Vararg{Int})::T
                    binoms = *([dynamic_binomial(k[i], idx[i]) for i in 1:nbr_vars]...)
                    ref_coeff = ps.ref.coefficients[scalar_idx][convertIndices_fullsym_to_lin(Tuple(k)...)]
                    centers = *([(ps.center[i] - ps.ref.center[i])^(k[i]-idx[i]) for i in 1:nbr_vars]...)
                    binoms*ref_coeff*centers
                end
                coeff = apply_with_fullsym_indices_from_and_upto(term1, +, trunc_order, idx...)
                ps.coefficients[scalar_idx][convertIndices_fullsym_to_lin(idx...)] = coeff
            else
                # construct function to compute the coefficients
                function term2(k::Vararg{Int})::T
                    binoms = *([binomial(k[i], idx[i]) for i in 1:nbr_vars]...)
                    ref_coeff = ps.ref.coefficients[scalar_idx][convertIndices_fullsym_to_lin(Tuple(k)...)]
                    centers = *([(ps.center[i] - ps.ref.center[i])^(k[i]-idx[i]) for i in 1:nbr_vars]...)
                    binoms*ref_coeff*centers
                end
                coeff = apply_with_fullsym_indices_from_and_upto(term2, +, trunc_order, idx...)
                ps.coefficients[scalar_idx][convertIndices_fullsym_to_lin(idx...)] = coeff
            end
        end
    end

    # change order
    ps.order = N

end


#--------------------------------------------Series defined by Partial Differential Equations----------------------------------------------

"""
    getSymbolics(x)::Number

    A function to retrieve the Symbolics representation of an object x. By default, assumes x isa
    Number. Using this function, one gets the fully algebraic representation of x
"""
getSymbolics(x::Number)::Number = x
getSymbolics(::Nothing) = nothing
getSymbolics(sc::SeriesCoefficient)::Number = sc.unique_sym

"""
    getNum(x)::Number

    A function to return the Number representing an object x. By default, assumes x isa
    Number. Using this function, objects that can be evaluated numerically are evaluated
    numerically
"""
getNum(x::Number)::Number = x
getNum(::Nothing) = nothing
function getNum(sc::SeriesCoefficient)::Number
    if sc.ps != :self
        if sc.ps.order < sc.indices[1]
            throw(ArgumentError("Cannot evaluate a SeriesCoefficient at order above the \
                                 currently computed order. Involved SeriesCoefficient : 
                                 $sc"))
        else
            sc.ps.coefficients[sc.index...][convertIndices_trunc_to_lin(sc.indices...)]
        end
    else
        getSymbolics(sc)
    end
end



############################ NlinearSeriesOperation ###########################
"""
    NlinearSeriesOperation

    SymbolicSeries are constructed as trees of SymbolicSeries. branches of that tree can
    be described as NlinearSeriesOperation. These operations correspond to the application
    of a func over a Vector args of arguments that can be SymbolicSeries, Numbers, ...

    ###Fields

    - `func` -- The function used to aggregate all the arguments. Should take a vector as 
      input.
    - `args::Vector` -- The Vector of arguments

    ###Example

    - `NlinearSeriesOperation(func, args::Vector)` -- default constructor
"""
struct NlinearSeriesOperation
    func
    args::Vector
end

function Base.show(io::IO, op::NlinearSeriesOperation)
    print(io, "NlinearSeriesOperation {$(op.func)}($(op.args))")
end

"""
    getSymbolics(op::NlinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the NlinearSeriesOperation
    Only works if op is an operation on SymbolicSeries coefficients.
"""
function getSymbolics(op::NlinearSeriesOperation)::Number
    to_aggregate = []
    for arg in op.args
        if arg isa ScalarSeriesSymbol
            throw(ArgumentError("Cannot get Symbolic expression of a ScalarSeriesSymbol."))
        else            
            push!(to_aggregate, getSymbolics(arg))
        end
    end
    getSymbolics(op.func(to_aggregate))
end



"""
    getNum(op::NlinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the NlinearSeriesOperation
    Only works if op is an operation on SymbolicSeries coefficients.

    Using this function, objects that can be evaluated numerically are evaluated 
    numerically
"""
function getNum(op::NlinearSeriesOperation)::Number
    to_aggregate = []
    for arg in op.args
        if arg isa ScalarSeriesSymbol
            throw(ArgumentError("Cannot get Symbolic expression of a ScalarSeriesSymbol."))
        else
            push!(to_aggregate, getNum(arg))
        end
    end
    getNum(op.func(to_aggregate))
end


################################## MultilinearSeriesOperation ##############################
"""
    MultilinearSeriesOperation

    SymbolicSeries are constructed as trees of SymbolicSeries. branches of that tree can
    be described as MultilinearSeriesOperation. These operations correspond to
    the application of a func over a range of arguments that is not necesarily defined at
    the time of the construction of the tree.

    ###Fiels

    - `aggreg` -- The function used to aggregate all the arguments. Should take two elements
      as input and make one
    - `arg` -- A function that returns the arguments to be used when called on a number of
      indices xᵢ. arg should return either a SeriesCoefficient, Numbers ... or an operation
      (NlinearSeriesOperation or MultilinearSeriesOperation) on SeriesCoefficients, Numbers 
      ... In particular, arg CANNOT return anything representing a Series.
    - `arg_parameters::Vector` -- parameters to be passed to the arg function. If not
      empty, arg should accept a named parameter `params`. If params can sometimes be
      empty, add the default value `params=[]`
    - `ranges::Vector{Tuple{Any, Any}}` -- The ranges between which the indices vary. i.e :
      xᵢ ∈ aᵢ:bᵢ where ranges = [(aᵢ, bᵢ)...]
    - `argNum2` -- Same as arg but should return the result of getNum2(arg(idcs...; params=arg_parameters), idx...; N). 
      where idcs is an element of ranges. Since idx and N cannot be known at the time of constructing the operation, 
      they must be passed to arg_parameters using :idxi and :∞ symbols

    ###Example

    - `MultilinearSeriesOperation(aggreg, 
                                  arg, 
                                  arg_parameters::Vector,
                                  ranges::Vector{Tuple{Any, Any}},
                                  argNum2)` -- default constructor
    - `MultilinearSeriesOperation(aggreg, 
                                  arg,
                                  ranges::Vector{Tuple{Any, Any}},
                                  argNum2)` -- constructor for 
      MultilinearSeriesOperation with empty arg_parameters
"""
struct MultilinearSeriesOperation
    aggreg
    arg
    arg_parameters::Vector
    ranges::Vector{Tuple{Any, Any}}
    argNum2
end

MultilinearSeriesOperation(aggreg, arg, ranges::Vector{Tuple{Any, Any}}, argNum2) = 
    MultilinearSeriesOperation(aggreg, arg, [], ranges, argNum2)

function Base.show(io::IO, op::MultilinearSeriesOperation)
    indices = ["i_$i, " for i in 1:length(op.ranges)]
    ranges_str = ["i_$i in $(r[1]):$(r[2]), " for (i,r) in enumerate(op.ranges)]
    print(io, "MultilinearSeriesOperation{$(op.aggreg)}(arg($(indices...); params=$(op.arg_parameters)) for $(ranges_str...))")
end

"""
    getSymbolics(op::MultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the MultilinearSeriesOperation.
    Only works if op is an operation on SymblicSeries coefficients.
"""
function getSymbolics(op::MultilinearSeriesOperation)::Number
    rngs = CartesianIndices(tuple(map(r -> r[1]:r[2], op.ranges)...))
    res = 0
    for idcs in rngs
        res = op.aggreg(res, getSymbolics(op.arg(Tuple(idcs)...; params=op.arg_parameters)))
    end
    res
end

"""
    getNum(op::MultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the MultilinearSeriesOperation.
    Only works if op is an operation on SymbolicSeries coefficients.

    Using this function, objects that can be evaluated numerically are evaluated 
    numerically
"""
function getNum(op::MultilinearSeriesOperation)::Number
    rngs = CartesianIndices(tuple(map(r -> r[1]:r[2], op.ranges)...))
    res = 0
    for idcs in rngs
        res = op.aggreg(res, getNum(op.arg(Tuple(idcs)...; params=op.arg_parameters)))
    end
    res
end

##############################TruncatedMultilinearSeriesOperation#########################
"""
    TruncatedMultilinearSeriesOperation

    TruncatedMultilinearSeriesOperation work similarly to MultilinearSeriesOperation
    but represent operations between an infinite number of terms, that have to be truncated
    at some order

    ###Fields

    - `aggreg` -- The function used to aggregate the terms of the operation. Must take two
      arguments and return one that can be reused by the function afterwards
    - `arg` -- The function used to generate the terms of the operation. Will be called
      with every possible index idx between start ∈ starts and +∞ (or rather the given
      truncation order). Must accept a named argument `params`
    - `arg_parameters::Vector` -- A vector of additionnal parameters that are passed to
      `arg` as the `params` named argument
    - `starts::Vector` -- The indices at which the indices should start
    - `maxOrder::Int` -- The order up to which these indices should go (can only be set by
      the getindex function and should be set to -1 by default)
    - `argNum2` -- Same as arg but should return the result of getNum2(arg(idcs...; params=arg_parameters), idx...; N). 
      where idcs is an element of ranges. Since idx and N cannot be known at the time of constructing the operation, 
      they must be passed to arg_parameters using :idxi and :∞ symbols
    
    ###Examples

    - `TruncatedMultilinearSeriesOperation(aggreg, 
                                           arg, 
                                           arg_parameters::Vector, 
                                           starts::Vector,
                                           argNum2)` -- default constructor
"""
struct TruncatedMultilinearSeriesOperation
    aggreg
    arg
    arg_parameters::Vector
    starts::Vector
    maxOrder::Int
    argNum2
end

TruncatedMultilinearSeriesOperation(aggreg, arg, arg_parameters::Vector, starts::Vector, argNum2) =
    TruncatedMultilinearSeriesOperation(aggreg, arg, arg_parameters, starts, -1, argNum2)

function Base.show(io::IO, op::TruncatedMultilinearSeriesOperation)
    starts = ["i_$i, " for i in 1:length(op.starts)]
    ranges_str = ["i_$i in $(r[1]):+∞, " for (i,r) in enumerate(op.starts)]
    print(io, "TruncatedMultilinearSeriesOperation{$(op.func)}(arg($(starts...); params=$(op.arg_parameters)) for $(ranges_str...))")
end

"""
    getSymbolics(op::TruncatedMultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the TruncatedMultilinearSeriesOperation.
    Only works if op is an operation on SymbolicSeries coefficients.
"""
function getSymbolics(op::TruncatedMultilinearSeriesOperation)::Number
    if op.maxOrder == -1
        throw("Cannot getSymbolic of a TruncatedMultilinearSeriesOperation that does not
               have a maxOrder")
    else
        apply_with_fullsym_indices_from_and_upto((k...) -> getSymbolics(op.arg(k...; params=op.arg_parameters)), 
                                                 op.aggreg,
                                                 op.maxOrder,
                                                 op.starts...)
    end
end

"""
    getNum(op::TruncatedMultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the TruncatedMultilinearSeriesOperation.
    Only works if op is an operation on SymbolicSeries coefficients.

    Using this function, objects that can be evaluated numerically are evaluated 
    numerically
"""
function getNum(op::TruncatedMultilinearSeriesOperation)::Number
    if op.maxOrder == -1
        throw("Cannot getNum of a TruncatedMultilinearSeriesOperation that does not
               have a maxOrder")
    else
        apply_with_fullsym_indices_from_and_upto((k...) -> getNum(op.arg(k...; params=op.arg_parameters)), 
                                                 op.aggreg,
                                                 op.maxOrder,
                                                 op.starts...)
    end
end

##################################### SymbolicSeries #####################################
"""
    {SymbolicSeries}{D}

    A representation of a series of D variables : ∑aᵢⱼₖxⁱyʲzᵏ for a 3 variables series.
    Throughout the tree of SymbolicSeries, the indices of the current node can be referred
    to as :idx1, :idx2, ..., :idxD.

    WARNING : The indexation convention is different from the convention used before.
              Indeed, to decide up to which order a series coefficient should be computed,
              it is easier to write them as a₀₀ + a₁₀ x + a₁₁ y + a₂₀ x² + ...
              However, operations between series if often easier to describe when 
              coefficients are written as ∑aᵢⱼₖxⁱyʲzᵏ. Thus, for SymbolicSeries (which are
              NOT a subtype of PowerSeries), this convention is adopted

    ### Fields

    - `ref::Union{ScalarSeriesSymbol, 
                  NlinearSeriesOperation,
                  MultilinearSeriesOperation,
                  TruncatedMultilinearSeriesOperation}` -- How are each coefficients defined.
      SymbolicSeries are represented as trees where SymbolicSeries are the nodes, 
      NlinearSeriesOperation, MultilinearSeriesOperation and TruncatedMultilinearSeriesOperation
      are the branches, and ScalarSeriesSymbol are the leaves (leaves may also be other objects such as 
      Numbers)
    - `center::Vector` -- A vector of length D which represents the center 
      around which the series should be computed when needed. Its component can be set two 
      :unspecified in which case if operations happen between this SymbolicSeries and
      another one, the other series specified center component will be used in priority
    - `get_selfseries_coefficients` -- A function which, given a Vararg{Int, D} returns the
      SeriesCoefficients that refer to series :self that appear when computing the
      coefficient of that series at this index. The return value should be a 
      Set{SeriesCoefficient}. This function can accept a named optionnal argument N which
      correspond to the order at which coefficients expression might be truncated if a 
      :∞ Symbol appears. (Used for LocalizedPDESeries)
    - `contains_selfseries::Int` -- Whether or not the SymbolicSeries' tree refers to the
      series :self at one point. If set to 0, it does not, if set to 1 it does, and if set
      to more than 1, then it refers to some derivatives of the series :self

    ### Examples

    - `SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                                 NlinearSeriesOperation, 
                                 MultilinearSeriesOperation,
                                 TruncatedMultilinearSeriesOperation}, 
                      center::Vector,
                      get_selfseries_coefficients,
                      contains_selfseries::Int)` -- default constructor
    - `SymbolicSeries(sss::ScalarSeriesSymbol, center::Vector)` -- Creates a SymbolicSeries
      from a ScalarSeriesSymbol
    - `SymbolicSeries(a::Array{ScalarSeriesSymbol}, center::Vector)::Array{SymbolicSeries}` 
      -- A constructor to easily create an array of SymbolicSeries{D} around the same 
      center
    - `SymbolicSeries(ps::PowerSeries)` -- A constructor to create a SymbolicSeries or an
      array of SymbolicSeries from a PowerSeries. If PowerSeries is scalar, then returns
      a SymbolicSeries. Otherwise returns an Array of SymbolicSeries. Center will then be 
      the center of the PowerSeries.

"""
struct SymbolicSeries{D}
    ref::Union{ScalarSeriesSymbol, NlinearSeriesOperation, MultilinearSeriesOperation, TruncatedMultilinearSeriesOperation}
    center::Vector
    get_selfseries_coefficients
    contains_selfseries::Int

    SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                              NlinearSeriesOperation,
                              MultilinearSeriesOperation,
                              TruncatedMultilinearSeriesOperation},
                   center::Vector,
                   get_selfseries_coefficients,
                   contains_selfseries::Int) = new{length(center)}(ref, 
                                                                    center, 
                                                                    get_selfseries_coefficients, 
                                                                    contains_selfseries)
end

function SymbolicSeries(sss::ScalarSeriesSymbol, center::Vector)
    if sss.ps != :self
        get_selfseries_coefficients1(I::Vararg{Int}; N=nothing) = Set()
        SymbolicSeries(sss, center, get_selfseries_coefficients1, 0)
    elseif sss.ps == :self
        get_selfseries_coefficients2(I::Vararg{Int}; N=nothing) = sss[convertIndices_fullsym_to_trunc(I...)...].ps == :self ? 
                                                                  Set([sss[convertIndices_fullsym_to_trunc(I...)...]]) : 
                                                                  Set()
        SymbolicSeries(sss, center, get_selfseries_coefficients2, 1)
    else
        throw(ArgumentError("Unknown symbol : $(sss.ps)"))
    end
end

function SymbolicSeries(a::Array{ScalarSeriesSymbol}, 
                        center::Vector)::Array{SymbolicSeries}
    map(sss -> SymbolicSeries(sss, center), a)
end

function SymbolicSeries(ps::PowerSeries)
    a = ps.scalar_series_ref
    length(a) == 1 ? SymbolicSeries(a[1], ps.center) : SymbolicSeries(a, ps.center)
end

# used for addition of constants
SymbolicSeries{D}(x::Number) where D = SymbolicSeries(NlinearSeriesOperation(v -> all(==(0), v) ? x : 0, generate_index_list(D)), fill(:unspecified, D), (I::Vararg{Int, D}; N=nothing) -> Set(), 0)
Base.convert(::Type{SymbolicSeries{D}}, x::Number) where D = SymbolicSeries{D}(x)

function Base.show(io::IO, s::SymbolicSeries)
    if s.ref isa ScalarSeriesSymbol
        if s.ref.ps isa PowerSeries
            print(io, "$(s.ref.ps.seriesID)$([s.ref.scalar_idx]...)")
        else 
            print(io, "$(s.ref.ps)$([s.ref.scalar_idx...])")
        end
    else
        print(io, s.ref)
    end
end

"""
    Base.getindex(s::SymbolicSeries{D}, args::Vararg{Any,D}) where D

    Returns a tree corresponding to the expression of the coefficient of s that has index
    args. This time, the nodes of the tree are NlinearSeriesOperations and 
    MultilinearSeriesOperations while the leaves are SeriesCoefficients.

    Named optionnal argument `N` is the truncation order when the coefficient expression
    must be truncated (i.e, some Symbol :∞ appears)

    ###Notes
    The actions performed by getindex on the SymbolicSeries tree are the following : 
    - substitutes ScalarSeriesSymbol with the SeriesCoefficient that corresponds to the given index
    - substitutes symbols :idx1, :idx2, ... with their corresponding values
    - substitutes SymbolicSeries with a call to getindex at the same index
    - applies the same operations recursively on NlinearSeriesOperation and MultilinearSeriesOperation
"""
function Base.getindex(s::SymbolicSeries{D}, args::Vararg{Any,D}; N=nothing) where D

    # truncation
    if !(isnothing(N)) && D > 0 && +(args...) > N
        return 0
    end

    # substitution
    idx_subst = Dict{Symbol, Union{Int, Nothing}}([Symbol("idx$i") => v for (i,v) in enumerate(args)])
    idx_subst[:∞] = N

    substitution2(t) = (substitution(t[1]), substitution(t[2]))
    function substitution(arg)
        if arg isa ScalarSeriesSymbol 
            arg[convertIndices_fullsym_to_trunc(args...)...]
        elseif arg isa SymbolicSeries
            arg[args..., N=N]
        elseif arg isa NlinearSeriesOperation
            NlinearSeriesOperation(arg.func, map(substitution, arg.args))
        elseif arg isa MultilinearSeriesOperation
            MultilinearSeriesOperation(arg.aggreg, 
                               arg.arg, 
                               substitution.(arg.arg_parameters), 
                               substitution2.(arg.ranges))
        elseif arg isa TruncatedMultilinearSeriesOperation
            isnothing(N) && throw(ArgumentError("Trying to getindex of a SymbolicSeries \
                                                 that depends on a TruncatedMultilinearSeriesOperation \
                                                 without giving named argument `N`"))
            TruncatedMultilinearSeriesOperation(arg.aggreg, 
                                                arg.arg,
                                                substitution.(arg.arg_parameters),
                                                substitution.(arg.starts),
                                                N)
        elseif arg in keys(idx_subst)
            idx_subst[arg]
        else
            arg
        end
    end
        
    substitution(s.ref)
end


"""
    getSymbolics(s::SymbolicSeries{D}, idx::Vararg{Int, D}) where D

    Returns the Symbolics expression of the coefficient of s at index idx

    ###Input

    - `s::SymbolicSeries{D}` -- A D-variables SymbolicSeries
    - `idx::Vararg{Int, D}` -- The index of the coefficient for which Symbolics expression
      should be computed

    ###Output

    The expression of the coefficient of s at index idx

    ###Note

    This is equivalent to getSymbolics(s[idx])
"""
getSymbolics(s::SymbolicSeries{D}, idx::Vararg{Int, D}) where D = getSymbolics(s[idx...])


"""
    getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}) where D

    Returns the Symbolics expression of the coefficient of s at index idx. Terms that can be
    evaluated numerically are evaluated numerically.

    ###Input

    - `s::SymbolicSeries{D}` -- A D-variables SymbolicSeries
    - `idx::Vararg{Int, D}` -- The index of the coefficient for which Symbolics expression
      should be computed

    ###Output

    The expression of the coefficient of s at index idx

    ###Note

    This is equivalent to getNum(s[idx])
"""
getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D = getNum(s[idx..., N=N])

Base.zero(::Type{SymbolicSeries{D}}) where D = SymbolicSeries(NlinearSeriesOperation(x -> 0, []), zeros(D), (I::Vararg{Int, D}; N=nothing) -> Set(), 0)
Base.zero(::SymbolicSeries{D}) where D = Base.zero(SymbolicSeries{D})


"""
    translate(s::SymbolicSeries{D}, new_center::Vector) where D

    Translates a SymbolicSeries at a different center

    ###Input

    - `s::SymbolicSeries{D}` -- The series to translate
    - `new_center::Vector` -- The new center of the series. Must have length D. If
      new center component or old center component is :unspecified, then it is assumed that
      it should not be translated

    ###Output

    A SymbolicSeries that has center `new_center` and returns the same values as `s` when
    evaluated numerically

    ###Notes

    The translation operation implies that the new coefficients depend on an infinite
    number of coefficients from the original series. Thus, any operation relying on this
    function cannot yield SymbolicSeriesEquation that can be used with simple PDESeries and
    instead has to make use of LocalizedPDESeries
"""
function translate(s::SymbolicSeries{D}, new_center::Vector) where D 
    # The `translate` function can be called even though N is not needed if some center'
    # components are undefined. Furthermore, some center components of the new center might
    # match those of the old center

    is_center_comp_trans = Bool[]
    res_center = []
    for (c, new_c) in zip(s.center, new_center)
        if c == new_c
            push!(is_center_comp_trans, false)
            push!(res_center, c)
        elseif c == :undefined
            push!(is_center_comp_trans, false)
            push!(res_center, new_c)
        elseif new_c == :undefined
            push!(is_center_comp_trans, false)
            push!(res_center, c)
        else # both defined and different
            push!(is_center_comp_trans, true)
            push!(res_center, new_c)
        end
    end

    # ensure that at least one center component has to be translated
    any(is_center_comp_trans) || return s

    function arg(k::Vararg{Int}; params=[])
        binoms = 1
        coeff_idx = []
        centers = 1
        implied_idx = 1
        for (i, is_implied, new_c, c) in 
            zip(1:D, is_center_comp_trans, new_center, s.center)

            if is_implied
                binoms *= dynamic_binomial(k[implied_idx], params[i])
                push!(coeff_idx, k[implied_idx])
                centers *= (new_c-c)^(k[implied_idx]-params[i])
                implied_idx += 1
            else
                push!(coeff_idx, params[i])
            end

        end
        NlinearSeriesOperation(v -> v[1]*v[2]*v[3], [binoms, 
                                                     s[coeff_idx..., N=params[end]], 
                                                     centers])
    end

    function argNum2(k::Vararg{Int}; params=[])
        binoms = 1
        coeff_idx = []
        centers = 1
        implied_idx = 1
        for (i, is_implied, new_c, c) in 
            zip(1:D, is_center_comp_trans, new_center, s.center)

            if is_implied
                binoms *= dynamic_binomial(k[implied_idx], params[i])
                push!(coeff_idx, k[implied_idx])
                centers *= (new_c-c)^(k[implied_idx]-params[i])
                implied_idx += 1
            else
                push!(coeff_idx, params[i])
            end

        end
        binoms*getNum2(s, coeff_idx...; N=params[end])*centers
    end
    
    idcs = generate_index_list(D)
    op = TruncatedMultilinearSeriesOperation(+, arg, [idcs; :∞], idcs[is_center_comp_trans], argNum2)

    function get_at_given_idx(I::NTuple{D, Int}, k::Vararg{Int}; N)
        coeff_idx = Vector{Int}(undef, D)
        implied_idx = 1
        for (idx, i, is_implied) in zip(1:D, I, is_center_comp_trans)
            if is_implied
                coeff_idx[idx] = k[implied_idx]
                implied_idx += 1
            else
                coeff_idx[idx] = i
            end
        end
        s.get_selfseries_coefficients(coeff_idx...; N=N)
    end

    get_selfseries_coefficients(I::Vararg{Int, D}; N) =
        apply_with_fullsym_indices_from_and_upto(k::Vararg{Int} -> get_at_given_idx(I, k...; N=N),
                                                 union,
                                                 N,
                                                 I[is_center_comp_trans]...)

    SymbolicSeries(op, res_center, get_selfseries_coefficients, s.contains_selfseries)
end

"""
    merge_centers(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D

    Make s1 and s2 have the same centers.

    ###Input

    - `s1::SymbolicSeries{D}` -- a SymbolicSeries of D variables
    - `s2::SymbolicSeries{D}` -- a SymbolicSeries of D variables

    ###Output

    new_s1, new_s2 -- two new SymbolicSeries that have the same center and behave exactly
    like s1 and s2 when evaluated numerically

    ###Notes

    The way the "center merging" operation is handled is not symetric: First, the center
    components that are specified in one of the series but not the other are transmitted
    to the other series. Then, if necessary, s2 is translated so that its center matches
    s1.
"""
function merge_centers(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D

    new_center1, new_center2 = [], []
    for (c1, c2) in zip(s1.center, s2.center)
        
        push!(new_center1, c1 == :unspecified ? c2 : c1)
        push!(new_center2, c2 == :unspecified ? c1 : c2)

    end

    new_s1 = SymbolicSeries(s1.ref, new_center1, s1.get_selfseries_coefficients, s1.contains_selfseries)
    new_s2 = SymbolicSeries(s2.ref, new_center2, s2.get_selfseries_coefficients, s2.contains_selfseries)

    if isequal(new_center1, new_center2) 
        (new_s1, new_s2)
    else
        if s2.contains_selfseries ≥ 1
            if s1.contains_selfseries ≥ 1
                if s2.contains_selfseries == 2 && s1.contains_selfseries == 2
                    @warn("A series depending on the derivatives of the selfseries \
                           had to be translated from center $new_center2 to $new_center1. \
                           This implies you will have to use LocalizedPDESeries and
                           computing the coefficients will most likely fail\
                           You may be able to avoid this translation by ensuring that the \
                           series with the resulting center is the left term of every operators. \
                           Please read the docs for more details")
                    (new_s1, translate(new_s2, new_center1))
                elseif s2.contains_selfseries == 2
                    @warn("A series depending on the :self series 
                           had to be translated from center $new_center1 to $new_center2. \
                           This implies you will have to use LocalizedPDESeries and can
                           increase the computational complexity of trying to compute the coefficients.\
                           You may be able to avoid this translation by ensuring that the \
                           series with the resulting center is the left term of every operators. \
                           Please read the docs for more details")
                    (translate(new_s1, new_center2), new_s2)
                else
                    @warn("A series depending on the :self series 
                           had to be translated from center $new_center2 to $new_center1. \
                           This implies you will have to use LocalizedPDESeries and can
                           increase the computational complexity of trying to compute the coefficients.\
                           You may be able to avoid this translation by ensuring that the \
                           series with the resulting center is the left term of every operators. \
                           Please read the docs for more details")
                    (new_s1, translate(new_s2, new_center1))
                end
            else
                @info("A series had to be translated from center $new_center1 to $new_center2. \
                   This implies you will have to use LocalizedPDESeries. \
                   You may be able to avoid this translation by ensuring that the \
                   series with the resulting center is the left term of every operators. \
                   Please read the docs for more details")
                (translate(new_s1, new_center2), new_s2)
            end
        else
            @info("A series had to be translated from center $new_center2 to $new_center1. \
                   This implies you will have to use LocalizedPDESeries. \
                   You may be able to avoid this translation by ensuring that the \
                   series with the resulting center is the left term of every operators. \
                   Please read the docs for more details")
            (new_s1, translate(new_s2, new_center1))
        end
    end

end

"""
    shift(s::SymbolicSeries{D}, by::NTuple{Int, D}) where D

    Shifts coefficients of `s` by `by`

    ###Input

    - `s::SymbolicSeries{D}` -- The series for which the coefficients must be shifted.
    - `by::NTuple{Int, D}` -- by how much the coefficients' indices should be shifted.
      Must be nonnegative

    ###Example

    If s(x,y) = ∑ᵢ,ⱼ sᵢⱼ xⁱyʲ, then shift(s, (1,2))(x,y) = ∑ᵢ,ⱼ s₍ᵢ₊₁₎₍ⱼ₊₂₎ xⁱyʲ
"""
function shift(s::SymbolicSeries{D}, by::NTuple{D, Int}) where D
    order_decrease = sum(by)
    op = NlinearSeriesOperation(v -> s[[v[i]+by[i] for i in 1:D]..., 
                                     N=isnothing(v[end]) ? nothing : v[end]-order_decrease],
                                [generate_index_list(D); :∞])

    get_selfseries_coefficients(idx::Vararg{Int, D}; N=nothing) = 
        s.get_selfseries_coefficients((idx .+ by)...; 
                                      N=isnothing(N) ? nothing : N-order_decrease)

    SymbolicSeries(op, s.center, get_selfseries_coefficients, s.contains_selfseries)
end

function Base.:+(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 + new_s2
    end

    op = NlinearSeriesOperation(v -> v[1] + v[2], [s1, s2])
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        s1.get_selfseries_coefficients(I...; N) ∪ s2.get_selfseries_coefficients(I...; N)
    SymbolicSeries(op, s1.center,  get_selfseries_coefficients, max(s1.contains_selfseries, s2.contains_selfseries))
end

function Base.:+(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    op = NlinearSeriesOperation(v -> all(==(0), v[1:D]) ? 
                                NlinearSeriesOperation(v -> v[1] + v[2], [s[v[1:D]..., N=v[end]], c[N=v[end]]]) : 
                                s[v[1:D]..., N=v[end]],
                                [generate_index_list(D); :∞])
    function get_selfseries_coefficients(idx::Vararg{Int, D}; N=nothing)
        if all(==(0), idx) && isequal(getNum(c), getNum(s, idx...))
            Set()
        else
            s.get_selfseries_coefficients(idx...; N=N) ∪ c.get_selfseries_coefficients(N=N)
        end
    end
    SymbolicSeries(op, s.center, get_selfseries_coefficients, s.contains_selfseries)
end
Base.:+(c::SymbolicSeries{0}, s::SymbolicSeries) = s+c

Base.:+(s::SymbolicSeries, x::Number) = s + SymbolicSeries{0}(x)
Base.:+(x::Number, s::SymbolicSeries) = SymbolicSeries{0}(x) + s

function Base.:-(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 - new_s2
    end

    op = NlinearSeriesOperation(v -> v[1] - v[2], [s1, s2])
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        s1.get_selfseries_coefficients(I...; N) ∪ s2.get_selfseries_coefficients(I...; N)
    SymbolicSeries(op, s1.center, get_selfseries_coefficients, max(s1.contains_selfseries, s2.contains_selfseries))
end


function Base.:-(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    op = NlinearSeriesOperation(v -> all(==(0), v[1:D]) ? 
                                NlinearSeriesOperation(v -> v[1] - v[2], [s[v[1:D]..., N=v[end]], c[N=v[end]]]) : 
                                s[v[1:D]..., N=v[end]],
                                [generate_index_list(D); :∞])
    function get_selfseries_coefficients(idx::Vararg{Int, D}; N=nothing) 
        if all(==(0), idx) && isequal(getNum(c), getNum(s, idx...))
            Set()
        else
            s.get_selfseries_coefficients(idx...; N=N) ∪ c.get_selfseries_coefficients(N=N)
        end
    end
    SymbolicSeries(op, s.center, get_selfseries_coefficients, s.contains_selfseries)
end
Base.:-(c::SymbolicSeries{0}, s::SymbolicSeries) = s-c

Base.:-(s::SymbolicSeries, x::Number) = s - SymbolicSeries{0}(x)
Base.:-(x::Number, s::SymbolicSeries) = SymbolicSeries{0}(x) - s

"""
    Base.:*(t::Number, s::SymbolicSeries)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:*(t::Number, s::SymbolicSeries)
    op = NlinearSeriesOperation(v -> v[1]*v[2], [t, s])
    SymbolicSeries(op, s.center, s.get_selfseries_coefficients, s.contains_selfseries)
end
"""
    Base.:*(s::SymbolicSeries, t::Number)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
Base.:*(s::SymbolicSeries, t::Number) = t*s
Base.:-(s::SymbolicSeries) = -1*s

"""
    Base.:/(s::SymbolicSeries, t::Number)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:/(s::SymbolicSeries, t::Number)
    op = NlinearSeriesOperation(v -> v[1]/v[2], [s, t])
    SymbolicSeries(op, s.center, s.get_selfseries_coefficients, s.contains_selfseries)
end

function Base.:*(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 * new_s2
    end

    arg(I::Vararg; params=[]) = NlinearSeriesOperation(v -> v[1]*v[2],
                                                       [s1[I..., N=params[end]], 
                                                        s2[(params[1:D].-I)..., N=params[end]]
                                                       ])

    argNum2(I::Vararg; params=[]) = getNum2(s1, I...; N=params[end])*getNum2(s2, I...; N=params[end])

    op = MultilinearSeriesOperation(+, arg, [generate_index_list(D); :∞], [(0, Symbol("idx$i")) for i in 1:D], argNum2)



    function get_self_series_coefficients(I::Vararg{Int, D}; N=nothing)
        ranges = CartesianIndices(tuple([0:i for i in I]...))
        res = Set()
        for idx in ranges
            if !isequal(getNum(s1[Tuple(idx)..., N=N]), 0)
                res = res ∪ s2.get_selfseries_coefficients((I.-Tuple(idx))...; N=N)
            end
            if !isequal(getNum(s2[(I.-Tuple(idx))..., N=N]), 0)
                res = res ∪ s1.get_selfseries_coefficients(Tuple(idx)...; N=N)
            end
        end
        res
    end

    SymbolicSeries(op, s1.center, get_self_series_coefficients, max(s1.contains_selfseries, s2.contains_selfseries))

end

##################################### EvaluatedSymbolicSeries #####################################
"""
    EvaluatedSymbolicSeries{D}

    This is an extension of SymbolicSeries{D} used for some operations such as integration, 
    derivation, or multiplication of series of multiple variables. 

    ### Fields

    - `series::SymbolicSeries{D}` -- The SymbolicSeries that is being evaluated
    - `variables::Vector{Num}` -- The variables in which the series is being evaluated

    ### Examples

    - `EvaluatedSymbolicSeries(series::SymbolicSeries, variables::Vector)` -- default 
      constructor
    - `EvaluatedSymbolicSeries(ps::PowerSeries)` -- if ps is a scalar PowerSeries, returns
      an EvaluatedSymbolicSeries. Otherwise, returns an array of EvaluatedSymbolicSeries.
    - `(s::SymbolicSeries{D})(at::Vararg{T,D})` -- 
      Constructs an EvaluatedSymbolicSeries by evaluating an existing SymbolicSeries in
      some variables or constants. 
        * variables : Num objects obtained by using the @variables macro from Symbolics.
                      Using operations on such variables can lead to unexpected results.
                      Only basic expressions such as x y z ... are supported.
                      One might use the same variables twice or more, but in that case, the
                      corresponding center components must all be the same if the user
                      whishes to compute it using a PDESeries. Otherwise, the coefficients
                      can only be computed using a LocalizedPDESeries. In this case, the
                      first center component will be used and the others disgarded.
        * constants : Can be the a component of the series' center. If so, the series can
                      then be computed using a PDESeries.
                      If the constant does not match series' center component, the series
                      will only be able to be computed using a LocalizedPDESeries.
    - `(s::Array{SymbolicSeries})(at::Vararg{Num})::Array{EvaluatedSymbolicSeries} -- 
      Constructs an array of EvaluatedSymbolicSeries by evaluating an array of
      SymbolicSeries
"""
struct EvaluatedSymbolicSeries{D}
    series::SymbolicSeries{D}
    variables::Vector{Num}

    function EvaluatedSymbolicSeries(series::SymbolicSeries{D}, variables::Vector{Num}) where D
        D == length(variables) || throw(ArgumentError("Cannot construct an EvaluatedSymbolicSeries \
                                                       for which some variables are not defined. \
                                                       (i.e : length(variables) does not match series  \
                                                        center size)"))
        new{D}(series, variables)
    end
end

function EvaluatedSymbolicSeries(ps::PowerSeries)
    if ps.size == (1,)
        EvaluatedSymbolicSeries(SymbolicSeries(ps), ps.variables)
    else
        map(s -> EvaluatedSymbolicSeries(s, ps.variables), SymbolicSeries(ps))
    end
end

Base.show(io::IO, ess::EvaluatedSymbolicSeries) = print(io, "$(ess.series)($(["$v," for v in ess.variables]...))")

"""
    getindex(ess::EvaluatedSymbolicSeries{D}, I::Vararg{Int, D}; N=nothing) where D

    Same as ess.series[I..., N=N]
"""
Base.getindex(ess::EvaluatedSymbolicSeries{D}, I::Vararg{Int, D}; N=nothing) where D = ess.series[I..., N=N]

Base.zero(ess::EvaluatedSymbolicSeries) = Base.zero(ess.series)(ess.variables...)

"""
    swap_variables(s::EvaluatedSymbolicSeries{D}, new_vars::Vector) where D

    Changes the order of s' variables to match the order given in new_vars

    ### Example

    If K(x,y,z) = ∑Kᵢⱼₖxⁱyʲzᵏ, then swapping variables of K for [y,z,x] will return a new
    EvaluatedSymbolicSeries K'(y,z,x) = ∑K'ᵢⱼₖyⁱzʲxᵏ where K'ᵢⱼₖ = Kₖᵢⱼ
"""
function swap_variables(s::EvaluatedSymbolicSeries{D}, new_vars::Vector) where D
    
    

    
    old_vars_idx = Dict([v => i for (i,v) in enumerate(s.variables)])
    new_vars_idx = Dict([v => i for (i,v) in enumerate(new_vars)])

    

    new_center = s.series.center[[old_vars_idx[v] for v in new_vars]]
    
    perm = [new_vars_idx[var] for var in s.variables]


    op = NlinearSeriesOperation(v -> s[v[perm]..., N=v[end]], 
                                [generate_index_list(D); :∞])

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        s.series.get_selfseries_coefficients(I[perm]..., N=N)

    SymbolicSeries(op, new_center, get_selfseries_coefficients, s.series.contains_selfseries)(new_vars...)
end


"""
    union_variables(s::EvaluatedSymbolicSeries, vars::Vector)

    Creates a new EvaluatedSymbolicSeries which variables are the union of s' variables
    and the variables from vars
"""
function union_variables(s::EvaluatedSymbolicSeries{D1}, vars::Vector) where D1

    new_vars = [s.variables; [setdiff(Set(vars), Set(s.variables))...]]
    D = length(new_vars)

    new_center = [s.series.center; fill(:unspecified, D-D1)]

    op = NlinearSeriesOperation(v -> all(==(0), v[D1+1:D]) ? s[v[1:D1]..., N=v[end]] : 0, 
                                [generate_index_list(D); :∞])

    get_selfseries_coefficients(I::Vararg{Int}; N=nothing) = all(==(0), I[D1+1:D]) ? 
        s.series.get_selfseries_coefficients(I[1:D1]...; N=N) : Set()

    SymbolicSeries(op, new_center, get_selfseries_coefficients, s.series.contains_selfseries)(new_vars...)
end


"""
    Base.:+(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries) 

    Sums two EvaluatedSymbolicSeries. If the center components of each series do not match, 
    then the use of LocalizedPDESeries is required instead of simple PDESeries
"""
function Base.:+(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    if isequal(s1.variables, s2.variables)
        (s1.series + s2.series)(s1.variables...)
    elseif isequal(Set(s1.variables), Set(s2.variables))
        s1 + swap_variables(s2, s1.variables)
    else
        union_variables(s1, s2.variables) + union_variables(s2, s1.variables)
    end

end

Base.:+(s1::SymbolicSeries{D}, s2::EvaluatedSymbolicSeries{D}) where D = s1(s2.variables...) + s2
Base.:+(s1::EvaluatedSymbolicSeries{D}, s2::SymbolicSeries{D}) where D = s1 + s2(s1.variables...)
Base.:+(s::EvaluatedSymbolicSeries{D}, x::Number) where D = s + (SymbolicSeries{D}(x))(s.variables...)
Base.:+(x::Number, s::EvaluatedSymbolicSeries{D}) where D = (SymbolicSeries{D}(x))(s.variables...) + s

Base.:+(s::EvaluatedSymbolicSeries, c::EvaluatedSymbolicSeries{0}) = (s.series+c.series)(s.variables...)
Base.:+(c::EvaluatedSymbolicSeries{0}, s::EvaluatedSymbolicSeries) = s+c
Base.:+(s::EvaluatedSymbolicSeries, c::SymbolicSeries{0}) = (s.series+c)(s.variables...)
Base.:+(c::SymbolicSeries{0}, s::EvaluatedSymbolicSeries) = s+c

"""
    Base.:-(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries) 
    
    Substracts two EvaluatedSymbolicSeries
"""
function Base.:-(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    if isequal(s1.variables, s2.variables)
        (s1.series - s2.series)(s1.variables...)
    elseif isequal(Set(s1.variables), Set(s2.variables))
        s1 - swap_variables(s2, s1.variables)
    else
        union_variables(s1, s2.variables) - union_variables(s2, s1.variables)
    end

end

Base.:-(s1::SymbolicSeries{D}, s2::EvaluatedSymbolicSeries{D}) where D = s1(s2.variables...) - s2
Base.:-(s1::EvaluatedSymbolicSeries{D}, s2::SymbolicSeries{D}) where D = s1 - s2(s1.variables...)
Base.:-(s::EvaluatedSymbolicSeries{D}, x::Number) where D = s - (SymbolicSeries{D}(x))(s.variables...)
Base.:-(x::Number, s::EvaluatedSymbolicSeries{D}) where D = (SymbolicSeries{D}(x))(s.variables...) - s

Base.:-(s::EvaluatedSymbolicSeries, c::EvaluatedSymbolicSeries{0}) = (s.series-c.series)(s.variables...)
Base.:-(c::EvaluatedSymbolicSeries{0}, s::EvaluatedSymbolicSeries) = s-c
Base.:-(s::EvaluatedSymbolicSeries, c::SymbolicSeries{0}) = (s.series-c)(s.variables...)
Base.:-(c::SymbolicSeries{0}, s::EvaluatedSymbolicSeries) = s-c

"""
    Base.:*(t::Number, s::EvaluatedSymbolicSeries)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:*(t::Number, s::EvaluatedSymbolicSeries)
    EvaluatedSymbolicSeries(t*s.series, s.variables)
end
"""
    Base.:*(s::EvaluatedSymbolicSeries, t::Number)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
Base.:*(s::EvaluatedSymbolicSeries, t::Number) = t*s
Base.:-(s::EvaluatedSymbolicSeries) = -1*s

"""
    Base.:/(s::EvaluatedSymbolicSeries, t::Number)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:/(s::EvaluatedSymbolicSeries, t::Number)
    EvaluatedSymbolicSeries(s.series/t, s.variables)
end

"""
    Base.:*(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    Multiply two EvaluatedSymbolicSeries. If some center components do not match between
    s1 and s2, then PDESeries cannot be used and LocalizedPDESeries must be used instead
"""
function Base.:*(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)
    
    if isequal(s1.variables, s2.variables)
        (s1.series * s2.series)(s1.variables...)
    elseif isequal(Set(s1.variables), Set(s2.variables))
        s1 * swap_variables(s2, s1.variables)
    else
        union_variables(s1, s2.variables) * union_variables(s2, s1.variables)
    end

end


function (d::Differential)(s::EvaluatedSymbolicSeries{D}) where D
    any([isequal(v, d.x) for v in s.variables]) || return zero(s)

    # find variable index
    x_idx = findfirst(v -> isequal(v, d.x), s.variables)
    
    # create index
    index = generate_index_list(D)

    # handle differentiation at order greater than 1
    new_s = d.order == 1 ? s : Differential(d.x, d.order-1)(s)
    
    op = NlinearSeriesOperation(u -> NlinearSeriesOperation(v -> v[1]*v[2], 
                                                            [(u[x_idx]+1), 
                                                             new_s.series[u[1:(x_idx-1)]..., u[x_idx]+1, u[(x_idx+1):D]..., N=(isnothing(u[end]) ? nothing : u[end]+1)]])
                                , [index; :∞])
    
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = new_s.series.get_selfseries_coefficients(I[1:(x_idx-1)]..., I[x_idx]+1, I[x_idx+1:end]...; (N = isnothing(N) ? N : N+1))
    EvaluatedSymbolicSeries(SymbolicSeries(op, 
                                           new_s.series.center, 
                                           get_selfseries_coefficients, 
                                           s.series.contains_selfseries ≥ 1 ? 2 : 0
                            ), new_s.variables)


end
(d::Differential)(a::Array{<:EvaluatedSymbolicSeries}) = d.(a)

"""
    ∫(s::EvaluatedSymbolicSeries{D}, x) where D

    Integrate s with respect to x (new constant coefficient is zero), meaning the
    integration is done from s' center to x.

    Throws an error if x is not one of s variables
"""
function ∫(s::EvaluatedSymbolicSeries{D}, x) where D
    any([isequal(v, x) for v in s.variables]) || throw(ArgumentError("$x is not a variable of $s. Available variables are : $(s.variables)"))

    # find variable index
    x_idx = findfirst(v -> isequal(v, x), s.variables)

    # create index
    index = generate_index_list(D)



    op = NlinearSeriesOperation(u -> NlinearSeriesOperation(v -> v[1] == 0 ? 0 : v[2]/v[1], 
                                                            [u[x_idx],
                                                             u[x_idx] == 0 ? 0 :  
                                                             s.series[u[1:(x_idx-1)]..., u[x_idx]-1, u[(x_idx+1):end-1]...,N=(isnothing(u[end]) ? nothing : u[end]-1)]
                                                            ])
                                , [index; :∞])

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = s.series.get_selfseries_coefficients(I[1:(x_idx-1)]..., I[x_idx]+1, I[x_idx+1:end]...; (N = isnothing(N) ? N : N-1))
    EvaluatedSymbolicSeries(SymbolicSeries(op, s.series.center, get_selfseries_coefficients, s.series.contains_selfseries), s.variables)

end
∫(a::Array{<:EvaluatedSymbolicSeries}, x) = map(s -> ∫(s, x), a)

"""
    ∫(s::EvaluatedSymbolicSeries, a, b, x)

    Integrate series s from a to b with respect to variable x

    ###Input

    - `s::EvaluatedSymbolicSeries` -- The series to integrate
    - `a` -- lower bound
    - `b` -- upper bound
    - `x` -- The variable with respect to which the series should be integrated. Throws
             an error if x is not among s variables
"""
function ∫(s::EvaluatedSymbolicSeries, a, b, x)
    # compute primitive
    prim = ∫(s, x)
    
    # compute bounds
    x_idx = findfirst(y -> isequal(x,y), prim.variables)
    up_bound = prim.variables[1:x_idx-1]..., b, prim.variables[x_idx+1:end]...
    low_bound = prim.variables[1:x_idx-1]..., a, prim.variables[x_idx+1:end]...

    # evaluate
    prim.series(up_bound...) - prim.series(low_bound...)
end
∫(arr::Array{<:EvaluatedSymbolicSeries}, a, b, x) = map(s -> ∫(s, a, b, x), arr)


########################## UnexpandedEvaluatedSymbolicSeries ########################

"""
    UnexpandedEvaluatedSymbolicSeries

    Similar to NlinearSeriesOperation, MultilinearSeriesOperation, or 
    TruncatedMultilinearSeriesOperation on SeriesCoefficients or ScalarSeriesSymbol,
    this is a wrapper for operations on EvaluatedSymbolicSeries, allowing the operation to
    not be expanded when writing the operation and instead, to be expanded later when
    calling the `expand` function.

    Its use is for instance relevant when composing power series, where the result will be 
    a truncated sum of powers of series, but the truncation order can only be known when 
    calling the `compute_coefficients!` method.

    All operations available on EvaluatedSymbolicSeries are also available on 
    UnexpandedEvaluatedSymbolicSeries

    ### Fields

    - `func` -- The function to be applied to the `arg` Vector. Arguments will be given as
      a Vector.
    - `args::Vector` -- The arguments to be given to `func`
    - `expansion_layer::Int` -- At which expansion layer was this
      UnexpandedEvaluatedSymbolicSeries created
    - `last_layer_func` -- Equivalent of func when the last expansion layer has been 
      reached
"""
struct UnexpandedEvaluatedSymbolicSeries
    func
    args::Vector
    expansion_layer::Int
    last_layer_func
end

get_expansion_layer(uess::UnexpandedEvaluatedSymbolicSeries) = uess.expansion_layer

function expand(uess::UnexpandedEvaluatedSymbolicSeries, N::Int)::EvaluatedSymbolicSeries
    if uess.expansion_layer ≥ N  
        res = uess.last_layer_func() 
        res isa UnexpandedEvaluatedSymbolicSeries ? expand(res, N) : res
    else
        expanded_args = []
        for arg in uess.args
            if arg isa UnexpandedEvaluatedSymbolicSeries
                push!(expanded_args, expand(arg, N))
            else
                push!(expanded_args, arg)
            end
        end
        res = uess.func(expanded_args)
        res isa UnexpandedEvaluatedSymbolicSeries ? expand(res, N) : res
    end
end

Base.zero(::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(x -> 0, [], 0, () -> 0)

unexp_add(v::Vector) = v[1] + v[2]
Base.:+(uess::UnexpandedEvaluatedSymbolicSeries, x) = UnexpandedEvaluatedSymbolicSeries(unexp_add, [uess, x], get_expansion_layer(uess), () -> uess.last_layer_func() + x)
Base.:+(x, uess::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_add, [x, uess], get_expansion_layer(uess), () -> uess.last_layer_func() + x)
Base.:+(uess1::UnexpandedEvaluatedSymbolicSeries, uess2::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_add, [uess1, uess2], max(get_expansion_layer(uess1), get_expansion_layer(uess2)), () -> uess1.last_layer_func() + uess2.last_layer_func())

unexp_sub(v::Vector) = v[1] - v[2]
Base.:-(uess::UnexpandedEvaluatedSymbolicSeries, x) = UnexpandedEvaluatedSymbolicSeries(unexp_sub, [uess, x], get_expansion_layer(uess), () -> uess.last_layer_func() - x)
Base.:-(x, uess::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_sub, [x, uess], get_expansion_layer(uess), () -> x - uess.last_layer_func())
Base.:-(uess1::UnexpandedEvaluatedSymbolicSeries, uess2::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_add, [uess1, uess2], max(get_expansion_layer(uess1), get_expansion_layer(uess2)), () -> uess1.last_layer_func() - uess2.last_layer_func())
Base.:-(uess::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_sub, [0, uess], get_expansion_layer(uess), () -> -uess.last_layer_func())

unexp_prod(v::Vector) = v[1] * v[2]
Base.:*(uess::UnexpandedEvaluatedSymbolicSeries, x) = UnexpandedEvaluatedSymbolicSeries(unexp_prod, [uess, x], get_expansion_layer(uess), () -> uess.last_layer_func() * x)
Base.:*(x, uess::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_prod, [x, uess], get_expansion_layer(uess), () -> uess.last_layer_func() * x)
Base.:*(uess1::UnexpandedEvaluatedSymbolicSeries, uess2::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_add, [uess1, uess2], max(get_expansion_layer(uess1), get_expansion_layer(uess2)), () -> uess1.last_layer_func() * uess2.last_layer_func())

unexp_div(v::Vector) = v[1] / v[2]
Base.:/(uess::UnexpandedEvaluatedSymbolicSeries, t::Number) = UnexpandedEvaluatedSymbolicSeries(unexp_div, [uess, t], get_expansion_layer(uess), () -> uess.last_layer_func() / t)

unexp_diff(v::Vector) = v[1](v[2])
(d::Differential)(uess::UnexpandedEvaluatedSymbolicSeries) = UnexpandedEvaluatedSymbolicSeries(unexp_diff, [d, uess], get_expansion_layer(uess), () -> d(uess.last_layer_func()))
(d::Differential)(a::Array{<:UnexpandedEvaluatedSymbolicSeries}) = map(uess -> d(uess), a)

unexp_prim(v::Vector) = ∫(v[1], v[2])
∫(uess::UnexpandedEvaluatedSymbolicSeries, x) = UnexpandedEvaluatedSymbolicSeries(unexp_prim, [uess, x], get_expansion_layer(uess), () -> ∫(uess.last_layer_func(), x))
∫(a::Array{<:UnexpandedEvaluatedSymbolicSeries}, x) = map(uess -> ∫(uess, x), a)

unexp_int(v::Vector) = ∫(v...)
∫(uess::UnexpandedEvaluatedSymbolicSeries, a, b, x) = UnexpandedEvaluatedSymbolicSeries(unexp_int, [uess, a, b, x], get_expansion_layer(uess), ∫(uess.last_layer_func(), a, b, x))
∫(arr::Array{<:UnexpandedEvaluatedSymbolicSeries}, a, b, x) = map(uess -> ∫(uess, a, b, x), arr)



############################# SymbolicSeries evaluation #############################


"""
    evaluate(s::SymbolicSeries{D}, at::Vararg{Any, D}) where D

    Evaluate `s` at `at` where elements of `at` can be either variables described by 
    Symbolics' `Num` objects, or constants.
"""
function evaluate(s::SymbolicSeries{D}, at::Vararg{Number, D}; _nbr_found=0) where D
    
    l = length(at)
    if l == _nbr_found # every argument has been used

        variables = Num[at...]
        return EvaluatedSymbolicSeries(s, variables)

    else

        x_idx = l-_nbr_found
        x = at[x_idx]
        if x isa Num # x is a variable

            other_x = findlast(v -> isequal(v, x), at[1:end-_nbr_found-1])
            
            if isnothing(other_x) # x isn't used as any other argument
                return evaluate(s, at...; _nbr_found=_nbr_found+1)
            
            else # x is also used somewhere else

                if (isequal(s.center[other_x], s.center[x_idx]) || 
                    s.center[other_x] == :unspecified || 
                    s.center[x_idx] == :unspecified) # the center components match => allow use of PDESeries

                    arg1(idx::Int; params) = s[params[1:other_x-1]..., 
                                              params[other_x]-idx, 
                                              params[other_x+1:x_idx-1]..., 
                                              idx, 
                                              params[x_idx:end-1]...,
                                              N=params[end]]

                    argNum2_1(idx::Int; params) = getNum2(s, params[1:other_x-1]..., 
                                                          params[other_x]-idx, 
                                                          params[other_x+1:x_idx-1]..., 
                                                          idx, 
                                                          params[x_idx:end-1]...; 
                                                          N=params[end])

                    op = MultilinearSeriesOperation(+, arg1, [generate_index_list(l-1); :∞], [(0,Symbol("idx$other_x"))], argNum2_1)

                    function get_selfseries_coefficients1(idcs::Vararg{Int}; N=nothing)
                        res = Set()
                        for idx in 0:idcs[other_x]
                            res = res ∪ s.get_selfseries_coefficients(
                                idcs[1:other_x-1]...,
                                idcs[other_x] - idx,
                                idcs[other_x+1:x_idx-1]...,
                                idx,
                                idcs[x_idx:end]...
                            ; N)
                        end
                        res
                    end

                    center = copy(s.center)
                    deleteat!(center, x_idx)
                    if center[other_x] == :unspecified 
                        center[other_x] = s.center[x_idx]
                    end

                    new_series = SymbolicSeries(op, center, get_selfseries_coefficients1, s.contains_selfseries)
                    new_at = at[1:x_idx-1]..., at[x_idx+1:end]...
                    return evaluate(new_series, new_at...; _nbr_found=_nbr_found)
                
                else # the center components do not match => LocalizedPDESeries

                    arg2(idx::Int; params) = NlinearSeriesOperation(sum, 
                                               [NlinearSeriesOperation(v -> v[1]*v[2]*v[3],
                                                [dynamic_binomial(idx, m),
                                                 s[
                                                    params[1:other_x-1]...,
                                                    params[other_x]-m,
                                                    params[other_x+1:x_idx-1]...,
                                                    idx,
                                                    params[x_idx:end-1]..., 
                                                    N=params[end]
                                                 ],
                                                 (s.center[other_x]-s.center[x_idx])^(idx-m)]                                               
                                               )                                                
                                               for m in 0:min(params[other_x],idx)
                                               ])

                    argNum2_2(idx::Int; params) = sum([dynamic_binomial(idx, m)*
                                                       getNum2(s, 
                                                               params[1:other_x-1]...,
                                                               params[other_x]-m,
                                                               params[other_x+1:x_idx-1]...,
                                                               idx,
                                                               params[x_idx:end-1]...;
                                                               N=params[end])*
                                                       (s.center[other_x]-s.center[x_idx])^(idx-m)
                                                       for m in 0:min(params[other_x], idx)])

                    op = MultilinearSeriesOperation(+, 
                                                    arg2, 
                                                    [generate_index_list(l-1);:∞], 
                                                    [(0,:∞)],
                                                    argNum2_2)
                    
                    function get_selfseries_coefficients2(idcs::Vararg{Int}; N::Int)
                        res = Set()
                        for idx in 0:N, m in 0:min(idcs[other_x],idx)
                            if +(idcs[1:other_x-1]..., idcs[other_x]-m, idcs[other_x+1:x_idx-1]...,
                                 idx, idcs[x_idx:end]...) ≤ N
                                res = res ∪ s.get_selfseries_coefficients(
                                    idcs[1:other_x-1]...,
                                    idcs[other_x]-m,
                                    idcs[other_x+1:x_idx-1]...,
                                    idx,
                                    idcs[x_idx:end]...
                                ; N)
                            end
                        end
                        res
                    end

                    center = copy(s.center)
                    deleteat!(center, x_idx)

                    new_series=SymbolicSeries(op, center, get_selfseries_coefficients2, s.contains_selfseries)
                    new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                    @info("You are evaluating a series with twice the same variables but \
                           center components do not match. Use LocalizedPDESeries to \
                           compute the coefficients.")

                    return evaluate(new_series, new_at...; _nbr_found=_nbr_found)
                    
                end

            end

        else # x is a constant

            if isequal(at[x_idx], s.center[x_idx]) # x matches center component => allow use of PDESeries

                index = generate_index_list(l-1)
                op = NlinearSeriesOperation(v -> s[v[1:x_idx-1]..., 0, v[x_idx:end-1]..., N=v[end]], 
                                            [index; :∞])
                
                get_selfseries_coefficients3(idcs::Vararg{Int}; N=nothing) = 
                    s.get_selfseries_coefficients(idcs[1:x_idx-1]...,0,idcs[x_idx:end]...; N)

                center = copy(s.center)
                deleteat!(center, x_idx)

                new_series = SymbolicSeries(op, center, get_selfseries_coefficients3, s.contains_selfseries)
                new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                return evaluate(new_series, new_at...; _nbr_found=_nbr_found)

            else # x does not match center component => LocalizedPDESeries

                arg4(idx; params) = NlinearSeriesOperation(v -> v[1]*v[2], 
                                                           [s[params[1:x_idx-1]...,idx,params[x_idx:end-1]...,N=params[end]],
                                                            (x-s.center[x_idx])^idx
                                                           ])
                
                argNum2_4(idx; params) = getNum2(s, params[1:x_idx-1]...,idx,params[x_idx:end-1]...;N=params[end])*
                                         (x-s.center[x_idx])^idx

                op = MultilinearSeriesOperation(+, arg4, [generate_index_list(l-1);:∞], [(0,:∞)], argNum2_4)

                function get_selfseries_coefficients4(idcs::Vararg{Int}; N)
                    res = Set()
                    for idx in 0:N
                        if +(idcs[1:x_idx-1]..., idx, idcs[x_idx:end]...) ≤ N
                            res = res ∪ s.get_selfseries_coefficients(idcs[1:x_idx-1]..., idx, idcs[x_idx:end]...; N)
                        end
                    end
                    res
                end

                center = copy(s.center)
                deleteat!(center, x_idx)

                new_series = SymbolicSeries(op, center, get_selfseries_coefficients4, s.contains_selfseries)
                new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                @info("You are evaluating a series at a point different from its center. \
                       Use LocalizedPDESeries to compute the coefficients.")

                return evaluate(new_series, new_at...; _nbr_found=_nbr_found)
            end

        end
    end

end

unexp_eval(v::Vector) = v[1](v[2]...; expansion_layer=v[6]+1) + v[3]*v[4](v[5]...; expansion_layer=v[6]+1)

"""
    evaluate_with_composition(s::SymbolicSeries{D}, at::Vararg{Any, D};
                              ess_idx::Int, expansion_layer::Int)::UnexpandedEvaluatedSymbolicSeries where D

    Evaluate `s` at `at` where elements of `at` can be either variables described by 
    Symbolics' `Num` objects, constants or SymbolicSeries. The result of this function is
    an UnexpandedEvaluatedSymbolicSeries, which allows composing series
"""
function evaluate_with_composition(s::SymbolicSeries{D}, at::Vararg{Any, D}; 
                                   ess_idx::Int, expansion_layer::Int)::UnexpandedEvaluatedSymbolicSeries where D
    
    # build new at to compute constant term
    new_at = [at...]
    new_at[ess_idx] = s.center[ess_idx]

    # build function to multiply other terms by
    f = at[ess_idx] - s.center[ess_idx]

    # build new series to evaluate
    by = zeros(Int, D)
    by[ess_idx] = 1
    new_s = shift(s, Tuple(by))

    # build resulting UnexpandedEvaluatedSymbolicSeries
    UnexpandedEvaluatedSymbolicSeries(unexp_eval, [s, new_at, f, new_s, at, expansion_layer], expansion_layer, () -> s(new_at...; expansion_layer=expansion_layer))
end

function (s::SymbolicSeries{D})(at::Vararg{Any, D}; expansion_layer=0) where D
    
    ess_idx = findfirst(x -> x isa EvaluatedSymbolicSeries, at)

    if isnothing(ess_idx)
        evaluate(s, at...)
    else
        evaluate_with_composition(s, at...; ess_idx, expansion_layer)
    end

end

function (a::Array{<:SymbolicSeries})(at::Vararg)::Array{EvaluatedSymbolicSeries}
    map(s -> s(at...), a)
end

################################ SymbolicSeriesEquation #############################

"""
    SymbolicSeriesEquation{D}

    A struct to represent the equality of two SymbolicSeries of D variables
    
    ###Fields 

    - `LHS::Union{SymbolicSeries{D}, Number}` -- Left Hand Side
    - `RHS::Union{SymbolicSeries{D}, Number}` -- Right Hand Side

    ###Examples

    - `SymbolicSeriesEquation(LHS::Union{SymbolicSeries{D}, Number}, 
                              RHS::Union{SymbolicSeries{D}, Number}) where D` -- 
      default constructor
    - `SymbolicSeriesEquation(LHS::Union{EvaluatedSymbolicSeries{D}, Number}, 
                              RHS::Union{EvaluatedSymbolicSeries{D}, Number}) where D` -- 
      same as default constructor but checks LHS and RHS have the same variables. 
      The order of their variables might differ
    - `Base.:~(LHS::Union{SymbolicSeries{D}, Number}, 
               RHS::Union{SymbolicSeries{D}, Number}) where D` -- a convenient shortcut 
      using binary operator ~
    - `Base.:~(LHS::Union{EvaluatedSymbolicSeries{D}, Number}, 
               RHS::Union{EvaluatedSymbolicSeries{D}, Number}) where D` -- same shortcut 
      for EvaluatedSymbolicSeries 
"""
struct SymbolicSeriesEquation{D}
    LHS::SymbolicSeries{D}
    RHS::SymbolicSeries{D}
end

get_nbr_vars(::SymbolicSeriesEquation{D}) where D = D

function SymbolicSeriesEquation(LHS::Union{EvaluatedSymbolicSeries{D1}, Number}, 
                                RHS::Union{EvaluatedSymbolicSeries{D2}, Number}) where {D1, D2}
    if LHS isa EvaluatedSymbolicSeries && RHS isa EvaluatedSymbolicSeries
        

        if isequal(LHS.variables, RHS.variables)
            if isequal(LHS.series.center, RHS.series.center)
                SymbolicSeriesEquation(LHS.series, RHS.series)
            else
                new_LHS, new_RHS = merge_centers(LHS.series, RHS.series)
                SymbolicSeriesEquation(new_LHS, new_RHS)
            end
        elseif isequal(Set(LHS.variables), Set(RHS.variables))
            SymbolicSeriesEquation(LHS, swap_variables(RHS, LHS.variables))
        else
            SymbolicSeriesEquation(union_variables(LHS, RHS.variables), union_variables(RHS, LHS.variables))
        end

    else
        SymbolicSeriesEquation(LHS isa EvaluatedSymbolicSeries ? LHS.series : SymbolicSeries{D2}(LHS), 
                               RHS isa EvaluatedSymbolicSeries ? RHS.series : SymbolicSeries{D1}(RHS))
    end
end

Base.:~(LHS::Union{EvaluatedSymbolicSeries, Number}, RHS::Union{EvaluatedSymbolicSeries, Number}) = SymbolicSeriesEquation(LHS, RHS)
Base.:~(LHS::Number, RHS::Number) = Equation(LHS, RHS)

Base.show(io::IO, eq::SymbolicSeriesEquation) = print(io, "SymbolicSeriesEquation{\n $(eq.LHS) \n ~ \n $(eq.RHS) \n}")

getSymbolics(eq::SymbolicSeriesEquation{D}, idx::Vararg{Int, D}) where D = (eq.LHS isa Number ? eq.LHS : getSymbolics(eq.LHS, idx...)) ~ (eq.RHS isa Number ? eq.RHS : getSymbolics(eq.RHS, idx...))
getNum(eq::SymbolicSeriesEquation{D}, idx::Vararg{Int, D}; N=nothing) where D = (eq.LHS isa Number ? eq.LHS : getNum(eq.LHS, idx..., N=N)) ~ (eq.RHS isa Number ? eq.RHS : getNum(eq.RHS, idx..., N=N))

"""
    get_involved_selfseries_coefficients(eq::SymbolicSeriesEquation{D}, I::Vararg{Int, D}) where D

    Retrieves all SeriesCoefficients that refer to :self series in the equation at a given 
    index I. The output is a Set{SeriesCoefficient}
"""
function get_involved_selfseries_coefficients(eq::SymbolicSeriesEquation{D},
                                              I::Vararg{Int, D};
                                              N=nothing
                                              )::Set{SeriesCoefficient} where D 

    res = Set()
    if eq.LHS.contains_selfseries ≥ 1
        res = res ∪ eq.LHS.get_selfseries_coefficients(I..., N=N)
    end

    if eq.RHS.contains_selfseries ≥ 1
        res = res ∪ eq.RHS.get_selfseries_coefficients(I..., N=N)
    end

    return res

end

############################### cached getNum function ##############################

getNum2(x::Any, ::Vararg{Int}; N=nothing) = x # default behaviour
getNum2(sss::ScalarSeriesSymbol, idx::Vararg{Int}; N=nothing) = getNum(sss[convertIndices_fullsym_to_trunc(idx...)...])
getNum2(sc::SeriesCoefficient, idx::Vararg{Int}; N=nothing) = getNum(sc)

function getNum2(sym::Symbol, idx::Vararg{Int}; N=nothing)
    if sym == :∞
        return N
    else
        for (i, v) in enumerate(idx)
            if sym == Symbol("idx$i")
                return v
            end
        end
        throw(ArgumentError("Tried to getNum of unknown symbol: $sym"))
    end
end

getNum2_rg(t::Tuple{Any, Any}, idx::Vararg{Int}; N=nothing) = (getNum2(t[1], idx...; N):getNum2(t[2], idx...; N))

getNum2(op::NlinearSeriesOperation, idx::Vararg{Int}; N=nothing) =     
    getNum2(op.func(map(arg -> getNum2(arg, idx...; N), op.args)), idx...; N)

function getNum2(op::MultilinearSeriesOperation, idx::Vararg{Int}; N=nothing)
    
    rngs = CartesianIndices(tuple(map(rg -> getNum2_rg(rg, idx...; N), op.ranges)...))
    params = map(p -> getNum2(p, idx...; N), op.arg_parameters)
    res = 0
    for idcs in rngs
        res = op.aggreg(res, op.argNum2(Tuple(idcs)...; params))
    end
    res
end

function getNum2(op::TruncatedMultilinearSeriesOperation, idx::Vararg{Int}; N)
    params = map(p -> getNum2(p, idx...; N), op.arg_parameters)
    apply_with_fullsym_indices_from_and_upto((k...) -> op.argNum2(k...; params), 
                                              op.aggreg,
                                              N,
                                              op.starts...)
end

getNum2(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D = getNum2(s.ref, idx...; N)

const _num_cache = IdDict{Any, Dict{Any, Any}}()
function cached_getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D
    d = get!(() -> Dict{Any, Any}(), _num_cache, s)
    get!(d, idx) do
        getNum2(s, idx...; N)
    end
end

clear_num_cache() = empty!(_num_cache)

function cached_getNum(eq::SymbolicSeriesEquation, idx::Vararg{Int}; N=nothing)
    cached_getNum(eq.LHS, idx...; N) ~ cached_getNum(eq.RHS, idx...; N)
end

######################### UnexpandedSymbolicSeriesEquation #######################

"""
    UnexpandedSymbolicSeriesEquation

    The equivalent of SymbolicSeriesEquation for UnexpandedEvaluatedSymbolicSeries. Can be
    used to create PDESeries or LocalizedPDESeries

    ### Fields
    
    `LHS::Union{UnexpandedEvaluatedSymbolicSeries, EvaluatedSymbolicSeries, SymbolicSeries, Number}`
    `RHS::Union{UnexpandedEvaluatedSymbolicSeries, EvaluatedSymbolicSeries, SymbolicSeries, Number}`
"""

struct UnexpandedSymbolicSeriesEquation
    LHS::Union{UnexpandedEvaluatedSymbolicSeries, EvaluatedSymbolicSeries, SymbolicSeries, Number}
    RHS::Union{UnexpandedEvaluatedSymbolicSeries, EvaluatedSymbolicSeries, SymbolicSeries, Number}
end

Base.:~(LHS::UnexpandedEvaluatedSymbolicSeries, RHS) = UnexpandedSymbolicSeriesEquation(LHS, RHS)
Base.:~(LHS, RHS::UnexpandedEvaluatedSymbolicSeries) = UnexpandedSymbolicSeriesEquation(LHS, RHS)
Base.:~(LHS::UnexpandedEvaluatedSymbolicSeries, RHS::UnexpandedEvaluatedSymbolicSeries) = UnexpandedSymbolicSeriesEquation(LHS, RHS)

function expand(eq::UnexpandedSymbolicSeriesEquation, N::Int)
    (eq.LHS isa UnexpandedEvaluatedSymbolicSeries ? expand(eq.LHS, N) : eq.LHS) ~
        (eq.RHS isa UnexpandedEvaluatedSymbolicSeries ? expand(eq.RHS, N) : eq.RHS)
end

expand(eq::SymbolicSeriesEquation, N::Int) = eq

################################# cached expand ##################################

const _expand_cache = IdDict{Any, Any}()
function cached_expand(uess::UnexpandedEvaluatedSymbolicSeries, N::Int)::EvaluatedSymbolicSeries
    get!(_expand_cache, uess) do
        if uess.expansion_layer ≥ N
            res = uess.last_layer_func()
            res isa UnexpandedEvaluatedSymbolicSeries ? expand(res, N) : res
        else
            expanded_args = []
            for arg in uess.args
                if arg isa UnexpandedEvaluatedSymbolicSeries
                    push!(expanded_args, cached_expand(arg, N))
                else
                    push!(expanded_args, arg)
                end
            end
            res = uess.func(expanded_args)
            res isa UnexpandedEvaluatedSymbolicSeries ? cached_expand(uess, N) : res
        end
    end
end

clear_expand_cache() = empty!(_expand_cache)

function cached_expand(eq::UnexpandedSymbolicSeriesEquation, N::Int)::SymbolicSeriesEquation
    LHS = eq.LHS isa UnexpandedEvaluatedSymbolicSeries ? cached_expand(eq.LHS, N) : eq.LHS
    RHS = eq.RHS isa UnexpandedEvaluatedSymbolicSeries ? cached_expand(eq.RHS, N) : eq.RHS
    LHS ~ RHS
end

cached_expand(eq::SymbolicSeriesEquation, N::Int)::SymbolicSeriesEquation = eq

################################### PDESeries ####################################

"""
    PDESeries{T,D} <: PowerSeries{T,D}

    A concrete type representing a PowerSeries defined by a PDE and its boundary conditions.
    The PDE system must be closed for it to work properly

    ### Fields

    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `maxIntegrationOrders::Array{ScalarSeriesSymbol, D}` -- An array of 
      ScalarSeriesSymbol that is used to easily create and access SeriesCoefficient

    - `equations::Vector{Union{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}}` 
      -- The PDE and its boundary conditions
    - `maxIntegrationOrders::Vector{Int}` -- What is the maximum number of consecutive
      integration operations of :self series that happen in each equation ? This will be 
      used to compute the maximum order at which each equation should be expanded. For 
      instance if one has an equation that does 2 integrations of the :self series, this
      value will be 2, and so all equations of orders up to order+2 will be expanded and 
      potentially used.
    - `used_equations::Vector{Set{Vector{Int}}}` -- For each equations, a set of the
      indices it has already been expanded to and solved for with previously computed
      orders

    ### Examples

    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}, 
                    equations::Vector{Union{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}},
                    maxIntegrationOrders::Vector{Int}) where T` -- default constructor
    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}, 
                    equations::Vector{Union{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}}) where T` 
      -- same as default constructor but maxIntegrationOrders default to 0
"""
mutable struct PDESeries{T,D} <: PowerSeries{T,D}
    
    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    scalar_series_ref::Array{ScalarSeriesSymbol, D}

    equations::Vector
    maxIntegrationOrders::Vector{Int}
    used_equations::Vector{Set{Vector{Int}}}

    function PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                          unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}, 
                          equations::Vector,
                          maxIntegrationOrders::Vector{Int}) where {T}

        _size = unknown isa AbstractArray ? size(unknown) : (1,)
        coefficients = map(x -> T[], CartesianIndices(_size))

        used_equations = []
        for _ in equations 
            push!(used_equations, Set{Vector{Int}}())
        end

        new{T, unknown isa AbstractArray ? ndims(unknown) : 1}(seriesID, _size, variables, 
        center, coefficients, -1, unknown isa AbstractArray ? unknown : [unknown], 
        equations, maxIntegrationOrders, used_equations)
    end

end

function PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector,
                      unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol},
                      equations::Vector) where T
    maxIntegrationOrders = zeros(Int, length(equations))
    PDESeries{T}(seriesID, variables, center, unknown, equations, maxIntegrationOrders)
end

"""
    expected_unknowns(K::ScalarSeriesSymbol, D::Int, N::Int)

    Generates the SeriesCoefficients of series K at order N

    ###Input

    - `K::ScalarSeriesSymbol` -- The ScalarSeriesSymbol from which the SeriesCoefficients
      will be created
    - `D::Int` -- The number of variables of K
    - `N::Int` -- The order for which to compute the SeriesCoefficient
"""
function expected_unknowns(K::ScalarSeriesSymbol, D::Int, N::Int)
    indices = generate_trunc_indices(N, D)
    map(idx -> K[idx...], indices)
end

expected_unknowns(K::Array{ScalarSeriesSymbol}, D::Int, N::Int) = map(sss -> expected_unknowns(sss, D, N), K)

"""
    PDES_should_discard_new_equation(unknowns::Set{<:ScalarSeriesSymbol}, N::Int)

    Based on the unknowns that appear in an equation, and the order for which it has been
    computed, whether or not to discard the equation. Equation should be discarded if 
      * unknowns Set is empty
      * coefficients of order > N appear in the unknowns
"""
function PDES_should_discard_new_equation(unknowns::Set{<:SeriesCoefficient}, N::Int)
    
    isempty(unknowns) && return true

    for uk in unknowns
        getOrder(uk) > N && return true
    end

    return false
end



"""
    compute_coefficients!(ps::PDESeries, N::Int; 
                          solver=symbolic_linear_solve, 
                          verbose::Int=0)

    Compute coefficients of ps up to order N.

    ###Input

    - `ps::PDESeries` -- the PowerSeries for which the coefficients should be computed
    - `N::Int` -- the order up to which the coefficients should be computed
    - `solver=julia_default` -- The solver to be used. The solver should take as 
      input a LinearSolve.LinearProblem and return a Vector of the coefficients values. 
      These values should be castable to the PDESeries T type parameter.
    - `verbose::Int=0` -- 
      * If verbose ≥ 1, indicates when an order is being computed and
      when it is done
      * If verbose ≥ 2, shows the equations and the unknowns for each order
      * If verbose ≥ 3, shows the discarded equations as well

    ###Output

    Stores new coefficients to ps, increases its order and changes the SeriesCoefficient
    that relate to series :self and whose value have been computed to relate to ps instead

    ###Notes

    - Since this function changes the PowerSeries the unknown relates to, one should not
      use the same unknown for different systems of equations. If required, this function
      and the use of its results can be encapsulated in a let ... end block to bypass this
      problem 
"""
function compute_coefficients!(ps::PDESeries{T}, N::Int; 
                               solver=julia_default, 
                               verbose::Int=0) where T

    N == -1 && (clear_num_cache(); clear_expand_cache())

    N ≤ ps.order && return
    
    N > ps.order+1 && compute_coefficients!(ps, N-1; solver=solver, verbose=verbose)
    

    verbose ≥ 1 && println("Computing coefficients of order $N")

    # first generate all expected unknowns of order up to N
    unknowns = expected_unknowns(ps.scalar_series_ref, length(ps.variables), N)
    nbr_coeffs = length(unknowns[1]) # will be used later
    unknowns = [unknowns...;]

    # expand UnexpandedSymbolicSeriesEquation if necessary
    ss_equations = map(eq -> cached_expand(eq, N), ps.equations)

    # then generate all equations of order N
    eqs = Equation[]
    for ((eq_idx, eq), maxIntOrd) in zip(enumerate(ss_equations), ps.maxIntegrationOrders)
        expand_for_indices = generate_fullsym_indices_upto(N+maxIntOrd, get_nbr_vars(eq))
        for idx in expand_for_indices
            if idx ∉ ps.used_equations[eq_idx]
                new_unknowns = get_involved_selfseries_coefficients(eq, idx...)
                if !(PDES_should_discard_new_equation(new_unknowns, N))
                    push!(eqs, cached_getNum(eq, idx...))
                    push!(ps.used_equations[eq_idx], idx)
                elseif verbose ≥ 3
                    println("discarded with fullsym index [$idx] : $(cached_getNum(eq, idx...))")
                    println("involved coefficients were $new_unknowns")
                end
            end
        end
    end

    if verbose ≥ 2
        println("Equations for order $N : ")
        foreach(eq -> println(eq), eqs)
        println("To solve with unknowns : ")
        foreach(unknown -> print("$unknown, "), unknowns)
        println()
    end

    # extract matrix A and vector b such that A.x = b where x is unknowns vector
    A, b = extract_affine_transformation(eqs, getSymbolics.(unknowns), T)

    # solve
    prob = LS.LinearProblem(A, b)
    res = solver(prob)

    # make room to add new coefficients 
    old_length = length(ps.coefficients[1])
    new_length = old_length + nbr_coeffs
    foreach(idx -> resize!(ps.coefficients[idx], new_length), eachindex(ps.coefficients))

    # fill in with the new coefficients
    for (sc, val) in zip(unknowns, res)
        ps.coefficients[sc.index...][convertIndices_trunc_to_lin(sc.indices...)...] = val
        sc.ps = ps
    end

    # update order
    ps.order = N
    verbose ≥ 1 && println("Coefficients computed up to order $N")

    return
end

################################# LocalizedPDESeries ###################################
"""
    LocalizedPDESeries{T,D} <: PowerSeries{T,D}

    A concrete type representing a PowerSeries defined by a PDE and its boundary conditions.
    The PDE system must be closed for it to work properly. 
        
    The main difference between
    PDESeries and LocalizedPDESeries comes from the compute_coefficients! method. 
    For PDESeries, this method computes the coefficients order by order. Thus the
    computation has a complexity of O(N³). For LocalizedPDESeries, all coefficients are 
    solved at the same time, making its complexity O(N⁴). However, this allows the 
    LocalizedPDESeries to use truncated symbolic series, for which the coefficient's
    expression is dependent on an infinite number of other coefficient, and thus has to be
    truncated to be computed.

    ### Fields

    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `scalar_series_ref::Array{ScalarSeriesSymbol, D}` -- An array of 
      ScalarSeriesSymbol that is used to easily create and access SeriesCoefficient

    - `equations::Vector{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}` -- The 
      PDE and its boundary conditions
    - `maxIntegrationOrders::Vector{Int}` -- What is the maximum number of consecutive
      integration operations of :self series that happen in each equation ? This will be 
      used to compute the maximum order at which each equation should be expanded. For 
      instance if one has an equation that does 2 integrations of the :self series, this
      value will be 2, and so all equations of orders up to order+2 will be expanded and 
      potentially used.


    ### Examples

    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    equations::Vector{Union{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}},
                    unknown::Union{Array{ScalarSeriesSymbol}, ScalarSeriesSymbol},
                    maxIntegrationOrders::Vector{Int}
                    ) where T` -- default constructor

    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    equations::Vector{Union{SymbolicSeriesEquation, UnexpandedSymbolicSeriesEquation}},
                    unknown::Union{Array{ScalarSeriesSymbol}, ScalarSeriesSymbol}) where T` 
      -- same as the default constructor but all maxIntegrationOrders are assumed to be 0.
"""
mutable struct LocalizedPDESeries{T,D} <: PowerSeries{T,D}
    
    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    scalar_series_ref::Array{ScalarSeriesSymbol, D}

    equations::Vector
    maxIntegrationOrders::Vector
    unknown::Array{ScalarSeriesSymbol, D}

    function LocalizedPDESeries{T}(seriesID::Symbol, variables::Vector{Num}, 
                                   center::Vector,
                                   scalar_series_ref::Array{<:ScalarSeriesSymbol, D},
                                   equations::Vector,
                                   maxIntegrationOrders::Vector{<:Int},
                                   unknown::Union{Array{<:ScalarSeriesSymbol}, 
                                                  ScalarSeriesSymbol}) where {T,D}

        _size = unknown isa AbstractArray ? size(unknown) : (1,)
        coefficients = map(x -> T[], CartesianIndices(_size))

        ps = new{T, unknown isa AbstractArray ? ndims(unknown) : 1}(seriesID, _size, variables, 
        center, coefficients, -1, scalar_series_ref,
        equations, maxIntegrationOrders, unknown isa AbstractArray ? unknown : [unknown])

        foreach(sss -> sss.ps=ps, scalar_series_ref)
        return ps
    end

end

function LocalizedPDESeries{T}(seriesID::Symbol, variables::Vector{Num}, 
                               center::Vector,
                               equations::Vector,
                               unknown::Union{Array{<:ScalarSeriesSymbol}, 
                                              ScalarSeriesSymbol},
                               maxIntegrationOrders::Vector{Int}) where T

    scalar_series_ref = map(idx -> ScalarSeriesSymbol(nothing, Tuple(idx), Dict()), unknown isa AbstractArray ? keys(unknown) : (1:1))

    LocalizedPDESeries{T}(seriesID, variables, center, scalar_series_ref, equations, maxIntegrationOrders, unknown)

end

function LocalizedPDESeries{T}(seriesID::Symbol, variables::Vector{Num},
                               center::Vector,
                               equations::Vector,
                               unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}) where T

    maxIntegrationOrders = zeros(Int, length(equations))
    LocalizedPDESeries{T}(seriesID, variables, center, equations, unknown, maxIntegrationOrders)
end

function expected_unknowns_upto(K::ScalarSeriesSymbol, D::Int, N::Int)
    indices = generate_trunc_indices_upto(N, D)
    map(idx -> K[idx...], indices)
end

expected_unknowns_upto(a::Array{<:ScalarSeriesSymbol}, D::Int, N::Int) = map(K -> expected_unknowns_upto(K, D, N), a)



"""
    compute_coefficients!(ps::LocalizedPDESeries, N::Int; 
                          solver=julia_default, verbose=0, benchmark::Bool=false)

    Compute coefficients of ps up to order N.

    ###Input

    - `ps::LocalizedPDESeries` -- the PowerSeries for which the coefficients should be computed
    - `N::Int` -- the order up to which the coefficients should be computed
    - `solver=QRFactorization()` -- The solver to be used. The solver should take as input
      a LinearSolve.LinearProblem and return a Vector of the coefficients values. These
      values should be castable to the LocalizedPDESeries T type parameter. It should also
      an optionnal named argument benchmark that can be used to display the time solving
      the system took
    - `verbose=0` -- verbose level
      * ≥ 1 -- print a message once the coefficients have been computed
      * ≥ 2 -- shows the equations that are to be solved and the associated unknowns
      * ≥ 3 -- shows the equations that were discarded
    - `benchmark::Bool=false` -- Used for benchmarking. If set to true, will print the
      time the solver took to solve the system of equations

    ###Output

    Stores new coefficients to ps and increases its order

"""
function compute_coefficients!(ps::LocalizedPDESeries{T}, N::Int; 
                               solver=julia_default, verbose=0, benchmark::Bool=false) where T

    clear_num_cache()
    clear_expand_cache()

    # first generate all expected unknowns of order up to N
    unknowns = expected_unknowns_upto(ps.unknown, length(ps.variables), N) 
    nbr_coeffs = length(unknowns[1]) # will be used later
    unknowns = [unknowns...;]

    # expand UnexpandedSymbolicSeriesEquation if necessary
    ss_equations = map(eq -> cached_expand(eq, N), ps.equations)

    # then generate all equations of orders up to N
    eqs = Equation[]
    for (eq, maxIntOrd) in zip(ss_equations, ps.maxIntegrationOrders)
        expand_for_indices = generate_fullsym_indices_upto(N+maxIntOrd, get_nbr_vars(eq))
        for idx in expand_for_indices
            new_unknowns = get_involved_selfseries_coefficients(eq, idx...; N=N)
            if !(PDES_should_discard_new_equation(new_unknowns, N))
                push!(eqs, cached_getNum(eq, idx...; N=N))
            elseif verbose ≥ 3
                println("discarded with fullsym index [$idx] : $(cached_getNum(eq, idx...; N=N)))")
                println("involved coefficients were $new_unknowns")
            end
        end
    end

    if verbose ≥ 2

        println("Equations : ")
        for eq in eqs
            println(eq)
        end

        println("To solve for unknowns : $unknowns")
    end

    # extract matrix A and vector b such that A.x = b where x is unknowns vector
    A, b = extract_affine_transformation(eqs, getSymbolics.(unknowns), T)

    # solve
    prob = LS.LinearProblem(A,b)
    t1 = time()
    res = solver(prob; benchmark=benchmark)
    t2 = time()
    if benchmark
        println("solver took $(t2-t1) seconds")
    end

    # make room to add new coefficients 
    foreach(idx -> resize!(ps.coefficients[idx], nbr_coeffs), eachindex(ps.coefficients))

    # fill in with the new coefficients
    for (sc, val) in zip(unknowns, res)
        ps.coefficients[sc.index...][convertIndices_trunc_to_lin(sc.indices...)...] = val
    end

    # update order
    ps.order = N
    verbose ≥ 1 && println("Coefficients computed up to order $N. Coefficients of higher order deleted")

end