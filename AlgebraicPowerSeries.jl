using Symbolics
import TaylorSeries
import LinearSolve as LS
include("utilitaries.jl")

#-------------------------------------------------------------PowerSeries--------------------------------------------------------
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
    u_sym, = @variables $u_sym

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
            sym, = @variables $sym
            sss.coefficients[vI] = SeriesCoefficient(:self, sym, vI, sss.scalar_idx)
        elseif sss.ps isa PowerSeries
            sym = Symbol("$(sss.ps.seriesID)$(sss.scalar_idx)$vI")
            sym, = @variables $sym
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
            # construct function to compute the coefficients
            function term(k::Vararg{Int})::T
                binoms = *([binomial(Int128(k[i]), idx[i]) for i in 1:nbr_vars]...)
                ref_coeff = ps.ref.coefficients[scalar_idx][convertIndices_fullsym_to_lin(Tuple(k)...)]
                centers = *([(ps.center[i] - ps.ref.center[i])^(k[i]-idx[i]) for i in 1:nbr_vars]...)
                binoms*ref_coeff*centers
            end
            coeff = apply_with_fullsym_indices_from_and_upto(term, +, trunc_order, idx...)
            ps.coefficients[scalar_idx][convertIndices_fullsym_to_lin(idx...)] = coeff
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
    be described as MultilinearSeriesOperation. These operations correspond to the
    the application of a func over a range of arguments that is not necesarily defined at
    the time of the construction of the tree. Typical use are the construction of 
    ExpandableFormula from the AlgebraicPowerSeries module

    ###Fiels

    - `func` -- The function used to aggregate all the arguments. Should take a vector as
      input
    - `arg` -- A function that returns the arguments to be used when called on a number of
      indices xᵢ. arg should return either a SeriesCoefficient, Numbers ... or an operation
      (NlinearSeriesOperation or MultilinearSeriesOperation) on SeriesCoefficients, Numbers 
      ... In particular, arg CANNOT return anything representing a Series.
    - `arg_parameters::Vector` -- parameters to be passed to the arg function. If not
      empty, arg should accept a named parameter `params`. If params can sometimes be
      empty, add the default value `params=[]`
    - `ranges::Vector{Tuple{Any, Any}}` -- The ranges between which the indices vary. i.e :
      xᵢ ∈ aᵢ:bᵢ where ranges = [(aᵢ, bᵢ)...]

    ###Example

    - `MultilinearSeriesOperation(func, 
                                  arg, 
                                  arg_parameters::Vector
                                  ranges::Vector{Tuple{Any, Any}})` -- default constructor
    - `MultilinearSeriesOperation(func, 
                                  arg,
                                  ranges::Vector{Tuple{Any, Any}})` -- constructor for 
      MultilinearSeriesOperation with empty arg_parameters
"""
struct MultilinearSeriesOperation
    func
    arg
    arg_parameters::Vector
    ranges::Vector{Tuple{Any, Any}}
end

MultilinearSeriesOperation(func, arg, ranges::Vector{Tuple{Any, Any}}) = 
    MultilinearSeriesOperation(func, arg, [], ranges)

function Base.show(io::IO, op::MultilinearSeriesOperation)
    indices = ["i_$i, " for i in 1:length(op.ranges)]
    ranges_str = ["i_$i in $(r[1]):$(r[2]), " for (i,r) in enumerate(op.ranges)]
    print(io, "MultilinearSeriesOperation{$(op.func)}(arg($(indices...); params=$(op.arg_parameters)) for $(ranges_str...))")
end

"""
    getSymbolics(op::MultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the MultilinearSeriesOperation.
    Only works if op is an operation on SymblicSeries coefficients.
"""
function getSymbolics(op::MultilinearSeriesOperation)::Number
    to_aggregate = []
    rngs = CartesianIndices(tuple(map(r -> r[1]:r[2], op.ranges)...))
    if isempty(op.arg_parameters)
        foreach(x -> push!(to_aggregate, getSymbolics(op.arg(Tuple(x)...))), rngs)
    else
        foreach(x -> push!(to_aggregate, 
                           getSymbolics(op.arg(Tuple(x)...; params=op.arg_parameters))),
                rngs)
    end
    getSymbolics(op.func(to_aggregate))
end

"""
    getNum(op::MultilinearSeriesOperation)::Number

    Returns the Symbolics.Num expression representing the MultilinearSeriesOperation.
    Only works if op is an operation on SymbolicSeries coefficients.

    Using this function, objects that can be evaluated numerically are evaluated 
    numerically
"""
function getNum(op::MultilinearSeriesOperation)::Number
    to_aggregate = []
    rngs = CartesianIndices(tuple(map(r -> r[1]:r[2], op.ranges)...))
    if isempty(op.arg_parameters)
        foreach(x -> push!(to_aggregate, getNum(op.arg(Tuple(x)...))), rngs)
    else
        foreach(x -> push!(to_aggregate, 
                           getNum(op.arg(Tuple(x)...; params=op.arg_parameters))),
                rngs)
    end
    getNum(op.func(to_aggregate))
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
                  MultilinearSeriesOperation}` -- How are each coefficients defined.
      SymbolicSeries are represented as trees where SymbolicSeries are the nodes, 
      NlinearSeriesOperation and MultilinearSeriesOperation are the branches,
      and ScalarSeriesSymbol are the leaves (leaves may also be other objects such as 
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

    ### Examples

    - `SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                                 NlinearSeriesOperation, 
                                 MultilinearSeriesOperation}, 
                      center::Vector,
                      get_selfseries_coefficients)` -- default constructor
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
    ref::Union{ScalarSeriesSymbol, NlinearSeriesOperation, MultilinearSeriesOperation}
    center::Vector
    get_selfseries_coefficients

    SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                              NlinearSeriesOperation,
                              MultilinearSeriesOperation},
                   center::Vector,
                   get_selfseries_coefficients) = new{length(center)}(ref, center, get_selfseries_coefficients)
end

function SymbolicSeries(sss::ScalarSeriesSymbol, center::Vector)
    if sss.ps != :self
        get_selfseries_coefficients1(I::Vararg{Int}; N=nothing) = Set()
        SymbolicSeries(sss, center, get_selfseries_coefficients1)
    else
        get_selfseries_coefficients2(I::Vararg{Int}; N=nothing) = sss[convertIndices_fullsym_to_trunc(I...)...].ps == :self ? 
                                                                  Set([sss[convertIndices_fullsym_to_trunc(I...)...]]) : 
                                                                  Set()
        SymbolicSeries(sss, center, get_selfseries_coefficients2)
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
SymbolicSeries{D}(x::Number) where D = SymbolicSeries(NlinearSeriesOperation(v -> all(==(0), v) ? x : 0, generate_index_list(D)), fill(:unspecified, D), (I::Vararg{Int, D}; N=nothing) -> Set())
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
            MultilinearSeriesOperation(arg.func, 
                               arg.arg, 
                               substitution.(arg.arg_parameters), 
                               substitution2.(arg.ranges))
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

Base.zero(::Type{SymbolicSeries{D}}) where D = SymbolicSeries(NlinearSeriesOperation(x -> 0, []), zeros(D), (I::Vararg{Int, D}; N=nothing) -> Set())
Base.zero(::SymbolicSeries{D}) where D = Base.zero(SymbolicSeries{D})

function Base.:+(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    center = []
    for (c1, c2) in zip(s1.center, s2.center)
        if c1 != :unspecified && c2 != :unspecified && c1!=c2
            throw(ArgumentError("Can only add SymbolicSeries that have the same centers"))
        elseif c1 == :unspecified
            push!(center, c2)
        else
            push!(center, c1)
        end
    end

    op = NlinearSeriesOperation(v -> v[1] + v[2], [s1, s2])
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        s1.get_selfseries_coefficients(I...; N) ∪ s2.get_selfseries_coefficients(I...; N)
    SymbolicSeries(op, center,  get_selfseries_coefficients)
end

Base.:+(s::SymbolicSeries{D}, x::Number) where D = s + SymbolicSeries{D}(x)
Base.:+(x::Number, s::SymbolicSeries{D}) where D = SymbolicSeries{D}(x) + s

function Base.:+(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    op = NlinearSeriesOperation(v -> all(==(0), v[1:D]) ? 
                                NlinearSeriesOperation(v -> v[1] + v[2], [s[v[1:D]..., N=v[end]], c[N=v[end]]]) : 
                                s[v[1:D]..., N=v[end]],
                                [generate_index_list(D); :∞])
    get_selfseries_coefficients(idx::Vararg{Int, D}; N=nothing) = s.get_selfseries_coefficients(idx...; N=N) ∪ c.get_selfseries_coefficients(N=N)
    SymbolicSeries(op, s.center, get_selfseries_coefficients)
end
Base.:+(c::SymbolicSeries{0}, s::SymbolicSeries) = s+c

function Base.:-(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    center = []
    for (c1, c2) in zip(s1.center, s2.center)
        if c1 != :unspecified && c2 != :unspecified && c1!=c2
            throw(ArgumentError("Can only substract SymbolicSeries that have the same centers"))
        elseif c1 == :unspecified
            push!(center, c2)
        else
            push!(center, c1)
        end
    end

    op = NlinearSeriesOperation(v -> v[1] - v[2], [s1, s2])
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        s1.get_selfseries_coefficients(I...; N) ∪ s2.get_selfseries_coefficients(I...; N)
    SymbolicSeries(op, center, get_selfseries_coefficients)
end

Base.:-(s::SymbolicSeries{D}, x::Number) where D = s - SymbolicSeries{D}(x)
Base.:-(x::Number, s::SymbolicSeries{D}) where D = SymbolicSeries{D}(x) - s

function Base.:-(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    op = NlinearSeriesOperation(v -> all(==(0), v[1:D]) ? 
                                NlinearSeriesOperation(v -> v[1] - v[2], [s[v[1:D]..., N=v[end]], c[N=v[end]]]) : 
                                s[v[1:D]..., N=v[end]],
                                [generate_index_list(D); :∞])
    get_selfseries_coefficients(idx::Vararg{Int, D}; N=nothing) = s.get_selfseries_coefficients(idx...; N=N) ∪ c.get_selfseries_coefficients(N=N)
    SymbolicSeries(op, s.center, get_selfseries_coefficients)
end
Base.:-(c::SymbolicSeries{0}, s::SymbolicSeries) = s-c

"""
    Base.:*(t::Number, s::SymbolicSeries)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:*(t::Number, s::SymbolicSeries)
    op = NlinearSeriesOperation(v -> v[1]*v[2], [t, s])
    SymbolicSeries(op, s.center, s.get_selfseries_coefficients)
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
    SymbolicSeries(op, s.center, s.get_selfseries_coefficients)
end

function Base.:*(s1::SymbolicSeries{1}, s2::SymbolicSeries{1})
    center = []
    for (c1, c2) in zip(s1.center, s2.center)
        if c1 != :unspecified && c2 != :unspecified && c1!=c2
            throw(ArgumentError("Can only multiply SymbolicSeries that have the same centers"))
        elseif c1 == :unspecified
            push!(center, c2)
        else
            push!(center, c1)
        end
    end
    
    arg(j; params::Vector=[]) = NlinearSeriesOperation(v -> v[1]*v[2], [s1[j,N=params[2]], s2[params[1]-j, N=params[2]]])
    op = MultilinearSeriesOperation(x -> Base.:+(x...), arg, [:idx1,:∞], [(0,:idx1)])
    get_selfseries_coefficients(i::Int; N=nothing) = union([s1.get_selfseries_coefficients(j;N) ∪ s2.get_selfseries_coefficients(i-j;N) for j in 0:i]...)
    SymbolicSeries(op, center, get_selfseries_coefficients)
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


function (s::SymbolicSeries{D})(at::Vararg{Any, D}; _nbr_found=0) where D
    
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
                return s(at...; _nbr_found=_nbr_found+1)
            
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
                    op = MultilinearSeriesOperation(v -> +(v...), arg1, [generate_index_list(l-1); :∞], [(0,Symbol("idx$other_x"))])

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

                    new_series = SymbolicSeries(op, center, get_selfseries_coefficients1)
                    new_at = at[1:x_idx-1]..., at[x_idx+1:end]...
                    return new_series(new_at...; _nbr_found=_nbr_found)
                
                else # the center components do not match => LocalizedPDESeries

                    arg2(idx::Int; params) = NlinearSeriesOperation(v -> +(v...), 
                                               [NlinearSeriesOperation(v -> v[1]*v[2]*v[3],
                                                [binomial(Int128(idx), m),
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
                    op = MultilinearSeriesOperation(v -> +(v...), 
                                                    arg2, 
                                                    [generate_index_list(l-1);:∞], 
                                                    [(0,:∞)])
                    
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

                    new_series=SymbolicSeries(op, center, get_selfseries_coefficients2)
                    new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                    return new_series(new_at...; _nbr_found=_nbr_found)
                    
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

                new_series = SymbolicSeries(op, center, get_selfseries_coefficients3)
                new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                return new_series(new_at...; _nbr_found=_nbr_found)

            else # x does not match center component => LocalizedPDESeries

                arg4(idx; params) = NlinearSeriesOperation(v -> v[1]*v[2], 
                                                           [s[params[1:x_idx-1]...,idx,params[x_idx:end-1]...,N=params[end]],
                                                            (x-s.center[x_idx])^idx
                                                           ])

                op = MultilinearSeriesOperation(v -> +(v...), arg4, [generate_index_list(l-1);:∞], [(0,:∞)])

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

                new_series = SymbolicSeries(op, center, get_selfseries_coefficients4)
                new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                return new_series(new_at...; _nbr_found=_nbr_found)
            end

        end
    end

end

function (a::Array{<:SymbolicSeries})(at::Vararg)::Array{EvaluatedSymbolicSeries}
    map(s -> s(at...), a)
end

Base.show(io::IO, ess::EvaluatedSymbolicSeries) = print(io, "$(ess.series)($(["$v," for v in ess.variables]...))")

"""
    getindex(ess::EvaluatedSymbolicSeries{D}, I::Vararg{Int, D}; N=nothing) where D

    Same as ess.series[I..., N=N]
"""
Base.getindex(ess::EvaluatedSymbolicSeries{D}, I::Vararg{Int, D}; N=nothing) where D = ess.series[I..., N=N]

Base.zero(ess::EvaluatedSymbolicSeries) = Base.zero(ess.series)(ess.variables...)

"""
    Base.:+(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    ###Notes

    Checks if s1 and s2 have the same variables. If their order differ, 
    the order of s1 is kept
"""
function Base.:+(s1::EvaluatedSymbolicSeries{D}, s2::EvaluatedSymbolicSeries{D}) where D
    # if necessary, swap indices of s2
    var_idx = Dict([v => i for (i,v) in enumerate(s1.variables)])
    index = generate_index_list(length(s1.variables))
    swap = NlinearSeriesOperation(index -> s2.series[[index[var_idx[v]] for v in s2.variables]...,N=index[end]], [index; :∞])
    get_selfseries_coefficients(index::Vararg{Int,D}; N=nothing) = s2.series.get_selfseries_coefficients([index[var_idx[v]] for v in s2.variables]...; N)
    new_s2series = SymbolicSeries(swap, s2.series.center, get_selfseries_coefficients)

    EvaluatedSymbolicSeries(s1.series + new_s2series, s1.variables)
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

    ###Notes

    Checks if s1 and s2 have the same variables. If their order differ, 
    the order of s1 is kept
"""
function Base.:-(s1::EvaluatedSymbolicSeries{D}, s2::EvaluatedSymbolicSeries{D}) where D
    isequal(Set(s1.variables), Set(s2.variables)) || throw(ArgumentError("Can only substract SymbolicSeries \
                                                         evaluated in the same variables"))

    # if necessary, swap indices of s2
    var_idx = Dict([v => i for (i,v) in enumerate(s1.variables)])
    index = generate_index_list(length(s1.variables))
    swap = NlinearSeriesOperation(index -> s2.series[[index[var_idx[v]] for v in s2.variables]...,N=index[end]], [index; :∞])
    get_selfseries_coefficients(index::Vararg{Int,D}; N=nothing) = s2.series.get_selfseries_coefficients([index[var_idx[v]] for v in s2.variables]...; N)
    new_s2series = SymbolicSeries(swap, s2.series.center, get_selfseries_coefficients)

    EvaluatedSymbolicSeries(s1.series - new_s2series, s1.variables)
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

    Multiply two EvaluatedSymbolicSeries. These two series might have different
    sets of variables and in different orders. The resulting EvaluatedSymbolicSeries
    will have the union of variables in the two series as variables and their order will
    be : order of the variables in the first series, then for variables that only appear in
    s2, order of the variables in s1. 
"""
function Base.:*(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)
    
    # associate each variable with its index in the vectors of variables
    variables_idx_s1 = Dict([v => i for (i,v) in enumerate(s1.variables)])
    variables_idx_s2 = Dict([v => i for (i,v) in enumerate(s2.variables)])

    # check if centers are the same
    for v in keys(variables_idx_s1) ∩ keys(variables_idx_s2)
        if (s1.series.center[variables_idx_s1[v]] != s2.series.center[variables_idx_s2[v]] &&
            s1.series.center[variables_idx_s1[v]] != :unspecified &&
            s2.series.center[variables_idx_s2[v]] != :unspecified)
            throw(ArgumentError("Center component $(s1.series.center[variables_idx_s1[v]]) \
                                 for variable $v in series $(s1.series) does not match \
                                 $(s2.series.center[variables_idx_s2[v]]) for variable \
                                 $v in series $(s2.series). Can't multiply"))
        end
    end

    # resulting variables
    res_variables = copy(s1.variables)
    added_variables_s2 = Num[setdiff(keys(variables_idx_s2), keys(variables_idx_s1))...]
    sort!(added_variables_s2, by=(v -> variables_idx_s2[v]))
    res_variables = [res_variables; added_variables_s2]
    res_variables_idx = Dict([v => i for (i,v) in enumerate(res_variables)])

    # construct center
    new_center = map( v -> v in keys(variables_idx_s1) ? 
                      ( s1.series.center[variables_idx_s1[v]] != :unspecified ?
                        s1.series.center[variables_idx_s1[v]] : 
                        ( v in keys(variables_idx_s2) ? 
                          s2.series.center[variables_idx_s2[v]] :
                          :unspecified
                        )
                      ) : 
                      s2.series.center[variables_idx_s2[v]],
                      res_variables
                    )

    # construct ranges
    common_variables = filter(v -> v in (keys(variables_idx_s1) ∩ keys(variables_idx_s2)),
                              res_variables)
    ranges = [(0, Symbol("idx$(res_variables_idx[v])")) for v in common_variables]
    ranges_idx = Dict([v => i for (i,v) in enumerate(common_variables)])

    # construct params
    params = generate_index_list(length(res_variables))
    idx_symbol_to_idx = Dict([s => i for (i,s) in enumerate(params)])
    params = [params;:∞]

    # construct arg
    common_variables_set = keys(variables_idx_s1) ∩ keys(variables_idx_s2)
    function arg(I::Vararg; params::Vector)
        ## construct coefficients indices
        idx_s1 = Vector(undef, length(s1.variables))
        idx_s2 = Vector(undef, length(s2.variables))
        for (i,v) in enumerate(s1.variables)
            idx_s1[i] = v in common_variables_set ? I[ranges_idx[v]] : params[res_variables_idx[v]]
        end
        for (i,v) in enumerate(s2.variables)
            idx_s2[i] = v in common_variables_set ? params[res_variables_idx[v]] - I[ranges_idx[v]] : params[res_variables_idx[v]]
        end

        ## return multiplication of the coefficients
        NlinearSeriesOperation(v -> v[1]*v[2], [s1.series[idx_s1...,N=params[end]], s2.series[idx_s2...,N=params[end]]])
    end

    # build get_selfseries_coefficients function
    function get_selfseries_coefficients(index::Vararg{Int}; N=nothing)
        # substitute values of index in ranges
        _ranges = map(t -> (t[1] in keys(idx_symbol_to_idx) ? index[idx_symbol_to_idx[t[1]]] : t[1]):
                          (t[2] in keys(idx_symbol_to_idx) ? index[idx_symbol_to_idx[t[2]]] : t[2]), 
                     ranges)
        
        res = Set()
        for I in CartesianIndices(tuple(_ranges...))
            idx_s1 = Vector(undef, length(s1.variables))
            idx_s2 = Vector(undef, length(s2.variables))
            for (i,v) in enumerate(s1.variables)
                idx_s1[i] = v in common_variables_set ? I[ranges_idx[v]] : index[res_variables_idx[v]]
            end
            for (i,v) in enumerate(s2.variables)
                idx_s2[i] = v in common_variables_set ? index[res_variables_idx[v]] - I[ranges_idx[v]] : index[res_variables_idx[v]]
            end
            res = res ∪ s1.series.get_selfseries_coefficients(idx_s1...; N) ∪ s2.series.get_selfseries_coefficients(idx_s2...; N)
        end


        res
    end

    # build resulting SymbolicSeries
    op = MultilinearSeriesOperation(v -> +(v...), arg, params, ranges)
    new_ss = SymbolicSeries(op, new_center, get_selfseries_coefficients)

    EvaluatedSymbolicSeries(new_ss, res_variables)
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
                                                             new_s.series[u[1:(x_idx-1)]..., u[x_idx]+1, u[(x_idx+1):D]...,N=(isnothing(u[end]) ? nothing : u[end]+1)]])
                                , [index; :∞])
    
    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = new_s.series.get_selfseries_coefficients(I[1:(x_idx-1)]..., I[x_idx]+1, I[x_idx+1:end]...; (N = isnothing(N) ? N : N+1))
    EvaluatedSymbolicSeries(SymbolicSeries(op, new_s.series.center, get_selfseries_coefficients), new_s.variables)


end
(d::Differential)(a::Array{<:EvaluatedSymbolicSeries}) = d.(a)

"""
    ∫(s::EvaluatedSymbolicSeries{D}, x) where D

    Integrate s with respect to x (new constant coefficient is zero), meaning the
    integration is done from s center to x.

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
    EvaluatedSymbolicSeries(SymbolicSeries(op, s.series.center, get_selfseries_coefficients), s.variables)

end
∫(a::Array{<:EvaluatedSymbolicSeries}, x) = map(s -> ∫(s, x), a)

"""
    ∫(s::EvaluatedSymbolicSeries, a, b, x)

    Integrate series s from a to b with respect to variable x

    ###Input

    - `s::EvaluatedSymbolicSeries` -- The series to integrate
    - `a` -- lower bound (must be a constant)
    - `b` -- upper bound (must be a constant)
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

"""
    ∫(s::EvaluatedSymbolicSeries{1}, a, b)

    Computes the integral of s from a to b

    ###Input

    - `s::EvaluatedSymbolicSeries{1}`
    - `a` -- The upper bound. Can be anything accepted by (::SymbolicSeries)(at...) but
      (::SymbolicSeries)(a) must return an EvaluatedSymbolicSeries of at most 1 variable
    - `b` -- The lower bound. Can be anything accepted by (::SymbolicSeries)(at...) but
      (::SymbolicSeries)(a) must return an EvaluatedSymbolicSeries of at most 1 variable

    ###Output
    
    An EvaluatedSymbolicSeries{D} where D = 0 or 1 depending on the input bounds.
    The center of this series is the same as s.
"""
function ∫(s::EvaluatedSymbolicSeries{1}, a, b)
    prim = (∫(s, s.variables[1])).series
    prim(b) - prim(a)
end
∫(arr::Array{<:EvaluatedSymbolicSeries{1}}, a, b) = map(s -> ∫(s, a, b), arr)


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
        variables_set = Set(LHS.variables) ∪ Set(RHS.variables)
        variables = [variables_set...]
        D = length(variables)

        # create resulting variables
        var_idx = Dict([v => i for (i,v) in enumerate(variables)])
        index = generate_index_list(D)

        # create resulting center
        LHS_var_idx = Dict([v => i for (i,v) in enumerate(LHS.variables)])
        RHS_var_idx = Dict([v => i for (i,v) in enumerate(RHS.variables)])
        center = []
        for v in variables
            if v ∉ keys(RHS_var_idx) || RHS_var_idx[v] == :unspecified
                push!(center, LHS.series.center[LHS_var_idx[v]])
            elseif v ∉ keys(LHS_var_idx) || LHS_var_idx[v] == :unspecified
                push!(center, RHS.series.center[RHS_var_idx[v]])
            elseif LHS.series.center[LHS_var_idx[v]] != RHS.series.center[RHS_var_idx[v]]
                throw(ArgumentError("Cannot create an equation between two series which
                                     centers do not match"))
            else
                push!(center, LHS.series.center[LHS_var_idx[v]])
            end
        end

        # swap indices to match array of resulting indices
        swap1 = NlinearSeriesOperation(index -> LHS.series[[index[var_idx[v]] for v in LHS.variables]...,N=index[end]], [index; :∞])
        get_selfseries_coefficients1(index::Vararg{Int}; N=nothing) = LHS.series.get_selfseries_coefficients([index[var_idx[v]] for v in LHS.variables]...; N)
        new_LHSseries = SymbolicSeries(swap1, center, get_selfseries_coefficients1)

        swap2 = NlinearSeriesOperation(index -> RHS.series[[index[var_idx[v]] for v in RHS.variables]...,N=index[end]], [index; :∞])
        get_selfseries_coefficients2(index::Vararg{Int}; N=nothing) = RHS.series.get_selfseries_coefficients([index[var_idx[v]] for v in RHS.variables]...; N)
        new_RHSseries = SymbolicSeries(swap2, center, get_selfseries_coefficients2)
        
        

        SymbolicSeriesEquation(new_LHSseries, new_RHSseries)
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
    (eq.LHS isa SymbolicSeries ? eq.LHS.get_selfseries_coefficients(I..., N=N) : Set()) ∪
    (eq.RHS isa SymbolicSeries ? eq.RHS.get_selfseries_coefficients(I..., N=N) : Set())
end

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

    - `equations::Vector{Union{Equation, SymbolicSeriesEquation}}` -- The PDE and its boundary 
      conditions
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
                    equations::Vector{Union{Equation, SymbolicSeriesEquation}},
                    maxIntegrationOrders::Vector{Int}) where T` -- default constructor
    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}, 
                    equations::Vector{Union{Equation, SymbolicSeriesEquation}}) where T` 
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

    equations::Vector{SymbolicSeriesEquation}
    maxIntegrationOrders::Vector{Int}
    used_equations::Vector{Set{Vector{Int}}}

    function PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                          unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}, 
                          equations::Vector{<:SymbolicSeriesEquation},
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
                      equations::Vector{SymbolicSeriesEquation}) where T
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
    - `solver=LS.QRFactorization()` -- The solver to be used. The solver should take as 
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
                               solver=LS.QRFactorization(), 
                               verbose::Int=0) where T

    N ≤ ps.order && return
    
    N > ps.order+1 && compute_coefficients!(ps, N-1; solver=solver, verbose=verbose)
    

    verbose ≥ 1 && println("Computing coefficients of order $N")

    # first generate all expected unknowns of order up to N
    unknowns = expected_unknowns(ps.scalar_series_ref, length(ps.variables), N)
    nbr_coeffs = length(unknowns[1]) # will be used later
    unknowns = [unknowns...;]

    # then generate all equations of order N
    eqs = Equation[]
    for ((eq_idx, eq), maxIntOrd) in zip(enumerate(ps.equations), ps.maxIntegrationOrders)
        expand_for_indices = generate_fullsym_indices_upto(N+maxIntOrd, get_nbr_vars(eq))
        for idx in expand_for_indices
            if idx ∉ ps.used_equations[eq_idx]
                new_unknowns = get_involved_selfseries_coefficients(eq, idx...)
                if !(PDES_should_discard_new_equation(new_unknowns, N))
                    push!(eqs, getNum(eq, idx...))
                    push!(ps.used_equations[eq_idx], idx)
                elseif verbose ≥ 3
                    println("discarded with fullsym index [$idx] : $(getNum(eq, idx...))")
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
    res = LS.solve(prob, solver)

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

    - `equations::Vector{SymbolicSeriesEquation}` -- The PDE and its boundary 
      conditions
    - `maxIntegrationOrders::Vector{Int}` -- What is the maximum number of consecutive
      integration operations of :self series that happen in each equation ? This will be 
      used to compute the maximum order at which each equation should be expanded. For 
      instance if one has an equation that does 2 integrations of the :self series, this
      value will be 2, and so all equations of orders up to order+2 will be expanded and 
      potentially used.


    ### Examples

    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    equations::Vector{Union{Equation, SymbolicSeriesEquation}},
                    unknown::Union{Array{ScalarSeriesSymbol}, ScalarSeriesSymbol},
                    maxIntegrationOrders::Vector{Int}
                    ) where T` -- default constructor

    - `PDESeries{T}(seriesID::Symbol, variables::Vector{Num}, center::Vector, 
                    equations::Vector{Union{Equation, SymbolicSeriesEquation}},
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

    equations::Vector{SymbolicSeriesEquation}
    maxIntegrationOrders::Vector
    unknown::Array{ScalarSeriesSymbol, D}

    function LocalizedPDESeries{T}(seriesID::Symbol, variables::Vector{Num}, 
                                   center::Vector,
                                   scalar_series_ref::Array{<:ScalarSeriesSymbol, D},
                                   equations::Vector{<:SymbolicSeriesEquation},
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
                               equations::Vector{<:SymbolicSeriesEquation},
                               unknown::Union{Array{<:ScalarSeriesSymbol}, 
                                              ScalarSeriesSymbol},
                               maxIntegrationOrders::Vector{Int}) where T

    scalar_series_ref = map(idx -> ScalarSeriesSymbol(nothing, Tuple(idx), Dict()), unknown isa AbstractArray ? keys(unknown) : (1:1))

    LocalizedPDESeries{T}(seriesID, variables, center, scalar_series_ref, equations, maxIntegrationOrders, unknown)

end

function LocalizedPDESeries{T}(seriesID::Symbol, variables::Vector{Num},
                               center::Vector,
                               equations::Vector{<:SymbolicSeriesEquation},
                               unknown::Union{Array{<:ScalarSeriesSymbol}, ScalarSeriesSymbol}) where T

    maxIntegrationOrders = zeros(Int, length(equations))
    LocalizedPDESeries{T}(seriesID, variables, center, equations, maxIntegrationOrders, unknown)
end

function expected_unknowns_upto(K::ScalarSeriesSymbol, D::Int, N::Int)
    indices = generate_trunc_indices_upto(N, D)
    map(idx -> K[idx...], indices)
end

expected_unknowns_upto(a::Array{<:ScalarSeriesSymbol}, D::Int, N::Int) = map(K -> expected_unknowns_upto(K, D, N), a)



"""
    compute_coefficients!(ps::LocalizedPDESeries, N::Int; 
                          solver=nothing, verbose=false)

    Compute coefficients of ps up to order N.

    ###Input

    - `ps::LocalizedPDESeries` -- the PowerSeries for which the coefficients should be computed
    - `N::Int` -- the order up to which the coefficients should be computed
    - `solver=QRFactorization()` -- The solver to be used. The solver should take as input
      a LinearSolve.LinearProblem and return a Vector of the coefficients values. These
      values should be castable to the LocalizedPDESeries T type parameter.
    - `verbose=0` -- verbose level
      * ≥ 1 -- print a message once the coefficients have been computed
      * ≥ 2 -- shows the equations that are to be solved and the associated unknowns
      * ≥ 3 -- shows the equations that were discarded

    ###Output

    Stores new coefficients to ps and increases its order

"""
function compute_coefficients!(ps::LocalizedPDESeries{T}, N::Int; solver=LS.QRFactorization(), verbose=0) where T

    # first generate all expected unknowns of order up to N
    unknowns = expected_unknowns_upto(ps.unknown, length(ps.variables), N) 
    nbr_coeffs = length(unknowns[1]) # will be used later
    unknowns = [unknowns...;]

    # then generate all equations of orders up to N
    eqs = Equation[]
    for (eq, maxIntOrd) in zip(ps.equations, ps.maxIntegrationOrders)
        expand_for_indices = generate_fullsym_indices_upto(N+maxIntOrd, get_nbr_vars(eq))
        for idx in expand_for_indices
            new_unknowns = get_involved_selfseries_coefficients(eq, idx...; N=N)
            if !(PDES_should_discard_new_equation(new_unknowns, N))
                push!(eqs, getNum(eq, idx...; N=N))
            elseif verbose ≥ 3
                println("discarded for fullsym idx [$idx] : $(getNum(eq, idx...; N=N))")
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
    res = LS.solve(prob, solver)

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