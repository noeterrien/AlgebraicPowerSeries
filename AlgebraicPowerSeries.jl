using Symbolics
import TaylorSeries
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
struct SeriesCoefficient{D}
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


"""
    getValue(sc::SeriesCoefficient)

    Returns the value of the coefficient. If it has not been computed yet, throws an error

    ###Input

    - `sc::SeriesCoefficient` -- a SeriesCoefficient

    ###Output

    If the SeriesCoefficient refers to a PowerSeries (and not :self), and the coefficient
    has already been computed, the value of this coefficient is returned.

    Otherwise, throws an error

"""
function getValue(sc::SeriesCoefficient)
    if sc.ps!=:self
        if sc.indices[1] ≤ sc.ps.order
            sc.ps.coefficients[sc.index...][convertIndices_trunc_to_lin(sc.indices...)]
        else
            throw(ArgumentError("Coefficient at index $(sc.indices) not computed yet. Cannot get value"))
        end
    else
        throw(ArgumentError("Cannot return value of a coefficient which refers to series \
                             :self"))
    end
end

#-----------------------------------------------------------SeriesSymbol-------------------------------------------------------------

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

    - `size::Vararg{Int}` -- The size of the series array

    ###Output

    An array of size size containing ScalarSeriesSymbol that refer to series :selfs
"""
function selfseries_symbols(size::Vararg{Int}) 
    ci = reshape(collect(CartesianIndices(size)), size)
    map(idx -> ScalarSeriesSymbol(:self, Tuple(idx), Dict()), ci)
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


#--------------------------------------------Series defined by Partial Differential Equations----------------------------------------------

"""
    getSymbolics(x)::Number

    A function to retrieve the Symbolics representation of an object x. By default, assumes x isa
    Number. Using this function, one gets the fully algebraic representation of x
"""
getSymbolics(x::Number)::Number = x
getSymbolics(sc::SeriesCoefficient)::Number = sc.unique_sym

"""
    getNum(x)::Number

    A function to return the Number representing an object x. By default, assumes x isa
    Number. Using this function, objects that can be evaluated numerically are evaluated
    numerically
"""
getNum(x::Number)::Number = x
function getNum(sc::SeriesCoefficient)::Number
    if sc.ps != :self
        if sc.ps.order < sc.indices[1]
            throw(ArgumentError("Cannot evaluate a SeriesCoefficient at order above the \
                                 currently computed order"))
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

    ###Fiels

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
    @variables x y
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
    SymbolicSeries{D}

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
    - `center::Vector` -- A vector of length D which represents the center around which
      the series should be computed when needed

    ### Examples

    - `SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                                 NlinearSeriesOperation, 
                                 MultilinearSeriesOperation}, 
                      center::Vector)` -- default constructor
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

    SymbolicSeries(ref::Union{ScalarSeriesSymbol, 
                              NlinearSeriesOperation,
                              MultilinearSeriesOperation},
                   center::Vector) = new{length(center)}(ref, center)
end

function SymbolicSeries(a::Array{ScalarSeriesSymbol}, 
                        center::Vector)::Array{SymbolicSeries}
    map(sss -> SymbolicSeries(sss, center), a)
end

function SymbolicSeries(ps::PowerSeries)
    a = ps.scalar_series_ref
    length(a) == 1 ? SymbolicSeries(a[1], ps.center) : SymbolicSeries(a, ps.center)
end

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

    ###Notes
    The actions performed by getindex on the SymbolicSeries tree are the following : 
    - substitutes ScalarSeriesSymbol with the SeriesCoefficient that corresponds to the given index
    - substitutes symbols :idx1, :idx2, ... with their corresponding values
    - substitutes SymbolicSeries with a call to getindex at the same index
    - applies the same operations recursively on NlinearSeriesOperation and MultilinearSeriesOperation
"""
function Base.getindex(s::SymbolicSeries{D}, args::Vararg{Any,D}) where D

    idx_subst = Dict([Symbol("idx$i") => v for (i,v) in enumerate(args)])
    substitution2(t) = (substitution(t[1]), substitution(t[2]))
    function substitution(arg) 
        if arg isa ScalarSeriesSymbol 
            arg[convertIndices_fullsym_to_trunc(args...)...]
        elseif arg isa SymbolicSeries
            arg[args...]
        elseif arg isa NlinearSeriesOperation
            NlinearSeriesOperation(arg.func, map(substitution, arg.args))
        elseif arg isa MultilinearSeriesOperation
            MultilinearSeriesOperation(arg.func, 
                               arg.arg, 
                               substitution.(s.ref.arg_parameters), 
                               substitution2.(s.ref.ranges))
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
getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}) where D = getNum(s[idx...])

function Base.:+(s1::SymbolicSeries, s2::SymbolicSeries)
    s1.center != s2.center && throw(ArgumentError("Can only add SymbolicSeries that have the \
                                                   same centers"))
    op = NlinearSeriesOperation(v -> v[1] + v[2], [s1, s2])
    SymbolicSeries(op, s1.center)
end

function Base.:-(s1::SymbolicSeries, s2::SymbolicSeries)
    s1.center != s2.center && throw(ArgumentError("Can only substract SymbolicSeries that have the \
                                                   same centers"))
    op = NlinearSeriesOperation(v -> v[1] + v[2], [s1, s2])
    SymbolicSeries(op, s1.center)
end

"""
    Base.:*(t::Number, s::SymbolicSeries)

    ###Notes

    WARNING : This operation is only supported when t has a numerical value.
    For instance, multiplication by a Symbolics' variable is non linear and
    will lead to unexpected results.
"""
function Base.:*(t::Number, s::SymbolicSeries)
    op = NlinearSeriesOperation(v -> v[1]*v[2], [t, s])
    SymbolicSeries(op, s.center)
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
    SymbolicSeries(op, s.center)
end

function Base.:*(s1::SymbolicSeries{1}, s2::SymbolicSeries{1})
    s1.center != s2.center && throw(ArgumentError("Can only multiply SymbolicSeries that have the same centers"))
    arg(j; params::Vector=[]) = NlinearSeriesOperation(v -> v[1]*v[2], [s1[j], s2[params[1]-j]])
    op = MultilinearSeriesOperation(x -> Base.:+(x...), arg, [:idx1], [(0,:idx1)])
    SymbolicSeries(op, s1.center)
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
    - `(s::SymbolicSeries{D})(at::Vararg{Num, D})::EvaluatedSymbolicSeries where D` -- 
      Constructs an EvaluatedSymbolicSeries by evaluating an existing SymbolicSeries in 
      some variables. Such variables can be obtained by using the Symbolics macro 
      @variables. Some variables can be the same if and only if the corresponding centers
      are equal.
    - `(s::SymbolicSeries{D})(at::Vararg{Union{Num, Symbol},
                                         D})::EvaluatedSymbolicSeries where D` -- Allows
      using Symbol :c to indicate that one wants to evaluate s at some of its center
      components. For instance, s(x, y, :c, t)
    - `(s::SymbolicSeries{D})(at::Vararg{Union{Num, Symbol, Float64},
                                         D})::EvaluatedSymbolicSeries where D` -- One
      might also want to write the center component value directly, which can be done
      using this method.      
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

function (s::SymbolicSeries{D})(at::Vararg{Num, D})::EvaluatedSymbolicSeries where D
    variables = Num[at...]
    
    # build a dictionnary that maps variables index to index of its first occurence
    # in the list
    found = Dict()
    index_mapper = Dict()
    for (i,v) in enumerate(variables)
        if v ∉ keys(found)
            found[v] = i
            index_mapper[i] = i
        else
            index_mapper[i] = found[v]
        end
    end

    # construct center and ensure identical variables have identical center components
    center = []
    for (i,v) in enumerate(variables)
        if i == index_mapper[i]
            push!(center, s.center[i])
        else
            s.center[i] == s.center[index_mapper[i]] || throw(ArgumentError("Series $s does not have the same \
                                                                             center components at index $i and \
                                                                             $(index_mapper[i])"))
        end
    end


    # compute new variables
    new_var = unique(variables)
    new_var_idx = Dict(v => i for (i,v) in enumerate(new_var))

    # construct params list
    params = generate_index_list(length(new_var))

    # construct ranges
    ranges = Tuple{Any, Any}[]
    for (i,v) in enumerate(variables)
        i != index_mapper[i] && push!(ranges, (0, params[new_var_idx[v]]))
    end

    # build arg function
    function arg(I::Vararg; params=[])
        idx = []
        rg_idx = 1
        for (i,v) in enumerate(variables)
            if i == index_mapper[i]
                push!(idx, params[new_var_idx[v]])
            else
                push!(idx, I[rg_idx])
                idx[index_mapper[i]] -= I[rg_idx]
                rg_idx += 1
            end
        end
        s[idx...]
    end

    # finally make new SymbolicSeries
    op = MultilinearSeriesOperation(v -> +(v...), arg, params, ranges)
    new_series = SymbolicSeries(op, center)

    EvaluatedSymbolicSeries(new_series, new_var)
end

function (s::SymbolicSeries{D})(at::Vararg{Union{Num, Symbol}, 
                                           D})::EvaluatedSymbolicSeries where D
    new_vars = []
    new_center = []
    var_idx = 1
    args = []
    for (i,v) in enumerate(at)
        if v == :c
            push!(args, 0)
        elseif v isa Num
            push!(new_vars, v)
            push!(new_center, s.center[i])
            push!(args, Symbol("idx$var_idx"))
            var_idx += 1
        else
            throw(ArgumentError("Unknown argument : $v"))
        end
    end

    op = NlinearSeriesOperation(v -> s[v...], args)
    new_series = SymbolicSeries(op, new_center)

    new_series(new_vars...)
end

function (s::SymbolicSeries{D})(at::Vararg{Union{Num, Symbol, AbstractFloat, Integer}, 
                                           D})::EvaluatedSymbolicSeries where D
    new_vars = []
    for (i,v) in enumerate(at)
        if v isa Union{AbstractFloat, Integer}
            v == s.center[i] || throw(ArgumentError("Trying to evaluate $s at a point that is not its origin"))
        push!(new_vars, :c)
        else
            push!(new_vars, v)
        end
    end

    s(new_vars...)
end

function (a::Array{SymbolicSeries})(at::Vararg)::Array{EvaluatedSymbolicSeries} where {SymbolicSeries}
    map(s -> s(at...), a)
end

Base.show(io::IO, ess::EvaluatedSymbolicSeries) = print(io, "$(ess.series)($(["$v," for v in ess.variables]...))")

Base.getindex(ess::EvaluatedSymbolicSeries{D}, I::Vararg{Int, D}) where D = ess.series[I...]


"""
    Base.:+(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    ###Notes

    Checks if s1 and s2 have the same variables. If their order differ, 
    the order of s1 is kept
"""
function Base.:+(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)
    isequal(Set(s1.variables), Set(s2.variables)) || throw(ArgumentError("Can only add SymbolicSeries \
                                                         evaluated in the same variables"))
    # if necessary, swap indices of s2
    var_idx = Dict([v => i for (i,v) in enumerate(s1.variables)])
    index = generate_index_list(length(s1.variables))
    swap = NlinearSeriesOperation(index -> s2.series[[index[var_idx[v]] for v in s2.variables]...], index)
    new_s2series = SymbolicSeries(swap, s2.series.center)

    EvaluatedSymbolicSeries(s1.series + new_s2series, s1.variables)
end


"""
    Base.:-(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)

    ###Notes

    Checks if s1 and s2 have the same variables. If their order differ, 
    the order of s1 is kept
"""
function Base.:-(s1::EvaluatedSymbolicSeries, s2::EvaluatedSymbolicSeries)
    isequal(Set(s1.variables), Set(s2.variables)) || throw(ArgumentError("Can only substract SymbolicSeries \
                                                         evaluated in the same variables"))

    # if necessary, swap indices of s2
    var_idx = Dict([v => i for (i,v) in enumerate(s1.variables)])
    index = generate_index_list(length(s1.variables))
    swap = NlinearSeriesOperation(index -> s2.series[[index[var_idx[v]] for v in s2.variables]...], index)
    new_s2series = SymbolicSeries(swap, s2.series.center)

    EvaluatedSymbolicSeries(s1.series - new_s2series, s1.variables)
end

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
        if s1.series.center[variables_idx_s1[v]] != s2.series.center[variables_idx_s2[v]]
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
    new_center = map(v -> v in keys(variables_idx_s1) ? 
                          s1.series.center[variables_idx_s1[v]] : 
                          s2.series.center[variables_idx_s2[v]],
                          res_variables)

    # construct ranges
    common_variables = filter(v -> v in (keys(variables_idx_s1) ∩ keys(variables_idx_s2)),
                              res_variables)
    ranges = [(0, Symbol("idx$(res_variables_idx[v])")) for v in common_variables]
    ranges_idx = Dict([v => i for (i,v) in enumerate(common_variables)])

    # construct params
    params = generate_index_list(length(res_variables))

    # construct arg
    common_variables_set = keys(variables_idx_s1) ∩ keys(variables_idx_s2)
    function arg(I::Vararg; params::Vector=[])
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
        NlinearSeriesOperation(v -> v[1]*v[2], [s1.series[idx_s1...], s2.series[idx_s2...]])
    end

    op = MultilinearSeriesOperation(v -> +(v...), arg, params, ranges)

    new_ss = SymbolicSeries(op, new_center)

    EvaluatedSymbolicSeries(new_ss, res_variables)
end

function (d::Differential)(s::EvaluatedSymbolicSeries{D}) where D
    # TODO : make it possible to return null series
    any([isequal(v, d.x) for v in s.variables]) || throw(ArgumentError("Trying to differentiate in variable $(d.x) \
                                                                        which is not amongst the variables of $s"))

    # find variable index
    x_idx = findfirst(v -> isequal(v, d.x), s.variables)

    # create index
    index = generate_index_list(D)

    op = NlinearSeriesOperation(u -> NlinearSeriesOperation(v -> v[1]*v[2], 
                                                            [(u[x_idx]+1), 
                                                             s.series[u[1:(x_idx-1)]..., u[x_idx]+1, u[(x_idx+1):end]...]])
                                , index)

    EvaluatedSymbolicSeries(SymbolicSeries(op, s.series.center), s.variables)

end
(d::Differential)(a::Array{EvaluatedSymbolicSeries}) = d.(a)



# TODO : make it possible to define PowerSeries with a function of N and previously 
# computed orders

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
    LHS::Union{SymbolicSeries{D}, Number}
    RHS::Union{SymbolicSeries{D}, Number}
end

function SymbolicSeriesEquation(LHS::Union{EvaluatedSymbolicSeries{D}, Number}, 
                                RHS::Union{EvaluatedSymbolicSeries{D}, Number}) where D
    if LHS isa EvaluatedSymbolicSeries && RHS isa EvaluatedSymbolicSeries
        isequal(Set(LHS.variables), Set(RHS.variables)) || throw(ArgumentError("Trying to construct a \
                                                           SymbolicSeriesEquation with two \
                                                           EvaluatedSymbolicSeries which \
                                                           variables do not match"))

        SymbolicSeriesEquation(LHS.series, RHS.series)
    else
        SymbolicSeriesEquation(LHS isa EvaluatedSymbolicSeries ? LHS.series : LHS, 
                               RHS isa EvaluatedSymbolicSeries ? RHS.series : RHS)
    end
end

Base.:~(LHS::Union{SymbolicSeries{D}, Number}, RHS::Union{SymbolicSeries{D}, Number}) where D = SymbolicSeriesEquation(LHS, RHS)
Base.:~(LHS::Union{EvaluatedSymbolicSeries{D}, Number}, RHS::Union{EvaluatedSymbolicSeries{D}, Number}) where D = SymbolicSeriesEquation(LHS, RHS)
Base.:~(LHS::Number, RHS::Number) = Equation(LHS, RHS)

Base.show(io::IO, eq::SymbolicSeriesEquation) = print(io, "SymbolicSeriesEquation{\n $(eq.LHS) \n ~ \n $(eq.RHS) \n}")

getSymbolics(eq::SymbolicSeriesEquation{D}, idx::Vararg{Int, D}) where D = getSymbolics(eq.LHS, idx...) ~ getSymbolics(eq.RHS, idx...)
getNum(eq::SymbolicSeriesEquation{D}, idx::Vararg{Int, D}) where D = getNum(eq.LHS, idx...) ~ getNum(eq.RHS, idx...)

