using Symbolics
import TaylorSeries
include("utils.jl")
include("solvers.jl")

#----------------------------------------- GLOBAL PARAMETERS ---------------------------------------------------------
_DISPLAY_INFO=true
_DISPLAY_WARN=true
set_display_info(to::Bool) = (_DISPLAY_INFO = to)
set_display_warn(to::Bool) = (_DISPLAY_WARN = to)

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

Base.show(io::IO, sss::ScalarSeriesSymbol) = print(io, "$(sss.ps isa Symbol ? sss.ps : sss.ps.seriesID)$(sss.scalar_idx)")


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

##################################### SymbolicSeries #####################################
"""
    {SymbolicSeries}{D}

    A representation of a series of D variables : ∑aᵢⱼₖxⁱyʲzᵏ for a 3 variables series.
    SymbolicSeries are represented as a tree. The nodes of that tree are SymbolicSeries and
    the leaves may be ScalarSeriesSymbol or Numbers.

    WARNING : The indexation convention is different from the convention used before.
              Indeed, to decide up to which order a series coefficient should be computed,
              it is easier to write them as a₀₀ + a₁₀ x + a₁₁ y + a₂₀ x² + ...
              However, operations between series if often easier to describe when 
              coefficients are written as ∑aᵢⱼₖxⁱyʲzᵏ. Thus, for SymbolicSeries (which are
              NOT a subtype of PowerSeries), this convention is adopted

    ### Fields

    - `center::Vector` -- A vector of length D which represents the center 
      around which the series should be computed when needed. Its component can be set two 
      :unspecified in which case if operations happen between this SymbolicSeries and
      another one, the other series specified center component will be used in priority
    - `getNum` -- A function which retrieves the numerical value of the series coefficient
      for a given index `idx::Vararg{Int}` and truncation order N. Its signature should be
      `getNum(idx::Vararg{Int, D}; N)`. If N is not mandatory, its default value should be 
      set to `nothing`
    - `get_selfseries_coefficients` -- A function which, given a Vararg{Int, D} returns the
      SeriesCoefficients that refer to series :self that appear when computing the
      coefficient of that series at this index. The return value should be a 
      Set{SeriesCoefficient}. This function can accept a named optionnal argument N which
      correspond to the order at which coefficients expression might be truncated if a 
      :∞ Symbol appears. (Used for LocalizedPDESeries)
    - `contains_selfseries::Int` -- Whether or not the SymbolicSeries' tree refers to the
      series :self at one point. If set to 0, it does not, if set to 1 it does, and if set
      to more than 1, then it refers to some derivatives of the series :self
    - `show` -- A function which is used to display the SymbolicSeries. Its signature
      should be `show()::String`
    - `should_cache` -- Whether or not the results of this operation should be cached when
      computing the coefficients

    ### Examples

    - `SymbolicSeries(getNum, 
                      center::Vector,
                      get_selfseries_coefficients,
                      contains_selfseries::Int,
                      show,
                      should_cache::Bool)` -- default constructor
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
    center::Vector
    getNum
    get_selfseries_coefficients
    contains_selfseries::Int
    show
    should_cache::Bool

    function SymbolicSeries(center::Vector, getNum, get_selfseries_coefficients,
                            contains_selfseries::Int, show, should_cache::Bool) 
        new{length(center)}(center, getNum, get_selfseries_coefficients, contains_selfseries, show, should_cache)
    end
end

#### Useful function when defining SymbolicSeries
get_selfseries_coefficients_none(::Int...; N=nothing) = Set()


function SymbolicSeries(sss::ScalarSeriesSymbol, center::Vector)

    getNum(I::Vararg{Int}; N=nothing) = cached_getNum(sss, I...; N)
    show() = "$(sss)"

    if sss.ps != :self
        SymbolicSeries(center, getNum, get_selfseries_coefficients_none, 0, show, true)

    elseif sss.ps == :self

        function get_selfseries_coefficients(I::Vararg{Int}; N=nothing) 
            sc = sss[convertIndices_fullsym_to_trunc(I...)...]
            if sc.ps == :self # check in case it has been moved to a PDESeries
                Set([sc])
            else 
                Set()
            end
        end

        SymbolicSeries(center, getNum, get_selfseries_coefficients, 1, show, true)

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
function SymbolicSeries{D}(x::Number) where D
    getNum(I::Int...; N=nothing) = all(==(0), I) ? x : 0
    show(I::Int...; N=nothing) = "$x"

    SymbolicSeries(fill(:unspecified, D), getNum, get_selfseries_coefficients_none, 0, show, false)
end
Base.convert(::Type{SymbolicSeries{D}}, x::Number) where D = SymbolicSeries{D}(x)

Base.show(io::IO, s::SymbolicSeries) = print(io, s.show())

Base.zero(::Type{SymbolicSeries{D}}) where D = SymbolicSeries{D}(0)
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

    function getNum(I::Vararg{Int, D}; N)
        function getNum_aux(k::Int...)
            binoms = 1
            coeff_idx = []
            centers = 1
            implied_idx = 1
            for (i, is_implied, new_c, c) in 
                zip(1:D, is_center_comp_trans, new_center, s.center)
                if is_implied
                    binoms *= dynamic_binomial(k[implied_idx], I[i])
                    push!(coeff_idx, k[implied_idx])
                    centers *= (new_c-c)^(k[implied_idx]-I[i])
                    implied_idx += 1
                else
                    push!(coeff_idx, I[i])
                end
            end
            binoms*cached_getNum(s, coeff_idx...; N)*centers
        end

        fixed_idcs = I[.!(is_center_comp_trans)]
        fixed_idcs_order = isempty(fixed_idcs) ? 0 : sum(fixed_idcs)
        apply_with_fullsym_indices_from_and_upto(getNum_aux, +, N-fixed_idcs_order, I[is_center_comp_trans]...)
    end


    function get_selfseries_coefficients(I::Vararg{Int, D}; N)
        function get_selfseries_coefficients_aux(k::Int...)
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
            cached_get_selfseries_coeffs(s, coeff_idx...; N=N)
        end

        apply_with_fullsym_indices_from_and_upto(get_selfseries_coefficients_aux,
                                                 union,
                                                 N-sum(I[.!(is_center_comp_trans)]),
                                                 I[is_center_comp_trans]...)
    end

    SymbolicSeries(res_center, getNum, get_selfseries_coefficients, s.contains_selfseries, s.show, true)
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

    new_s1 = SymbolicSeries(new_center1, s1.getNum, s1.get_selfseries_coefficients, s1.contains_selfseries, s1.show, false)
    new_s2 = SymbolicSeries(new_center2, s2.getNum, s2.get_selfseries_coefficients, s2.contains_selfseries, s2.show, false)

    if isequal(new_center1, new_center2) 
        (new_s1, new_s2)
    else
        if s2.contains_selfseries ≥ 1
            if s1.contains_selfseries ≥ 1
                if s2.contains_selfseries == 2 && s1.contains_selfseries == 2
                    _DISPLAY_WARN && 
                    @warn("A series depending on the derivatives of the selfseries \
                           had to be translated from center $new_center2 to $new_center1. \
                           This implies you will have to use LocalizedPDESeries and
                           computing the coefficients will most likely fail\
                           You may be able to avoid this translation by ensuring that the \
                           series with the resulting center is the left term of every operators. \
                           Please read the docs for more details")
                    (new_s1, translate(new_s2, new_center1))
                elseif s2.contains_selfseries == 2
                    _DISPLAY_WARN &&
                    @warn("A series depending on the :self series 
                           had to be translated from center $new_center1 to $new_center2. \
                           This implies you will have to use LocalizedPDESeries and can
                           increase the computational complexity of trying to compute the coefficients.\
                           You may be able to avoid this translation by ensuring that the \
                           series with the resulting center is the left term of every operators. \
                           Please read the docs for more details")
                    (translate(new_s1, new_center2), new_s2)
                else
                    _DISPLAY_WARN &&
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
                _DISPLAY_INFO &&
                @info("A series had to be translated from center $new_center1 to $new_center2. \
                   This implies you will have to use LocalizedPDESeries. \
                   You may be able to avoid this translation by ensuring that the \
                   series with the resulting center is the left term of every operators. \
                   Please read the docs for more details")
                (translate(new_s1, new_center2), new_s2)
            end
        else
            _DISPLAY_INFO &&
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
    
    getNum(I::Vararg{Int, D}; N=nothing) = cached_getNum(s, (I.+by)...; N = isnothing(N) ? nothing : N-order_decrease)

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) =
        cached_get_selfseries_coeffs(s, (I .+ by)...; N=isnothing(N) ? nothing : N-order_decrease)

    show() = "($s)↑$by"

    SymbolicSeries(s.center, getNum, get_selfseries_coefficients, s.contains_selfseries, show, false)
end

function Base.:+(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 + new_s2
    end

    getNum(I::Vararg{Int, D}; N=nothing) = cached_getNum(s1, I...; N) + cached_getNum(s2, I...; N)

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        cached_get_selfseries_coeffs(s1, I...; N) ∪ 
            cached_get_selfseries_coeffs(s2, I...; N)

    show() = "($s1 + $s2)"

    SymbolicSeries(s1.center, getNum, get_selfseries_coefficients, 
                   max(s1.contains_selfseries, s2.contains_selfseries), show, false)
end

function Base.:+(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    getNum(I::Vararg{Int, D}; N=nothing) = 
        all(==(0), I) ? cached_getNum(s, I...; N) + cached_getNum(c; N) : cached_getNum(s, I...; N)

    function get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing)
        if all(==(0), I) && isequal(-cached_getNum(c; N), cached_getNum(s, I...; N))
            Set()
        else
            cached_get_selfseries_coeffs(s, I...; N) ∪ cached_get_selfseries_coeffs(c; N)
        end
    end

    show() = "($s + $c)"

    SymbolicSeries(s.center, getNum, get_selfseries_coefficients, 
                   max(s.contains_selfseries, c.contains_selfseries), show, false)
end
Base.:+(c::SymbolicSeries{0}, s::SymbolicSeries) = s+c

function Base.:+(s1::SymbolicSeries{0}, s2::SymbolicSeries{0})

    getNum(;N=nothing) = cached_getNum(s1; N) + cached_getNum(s2; N)

    get_selfseries_coefficients(;N=nothing) = cached_get_selfseries_coeffs(s1; N) ∪ cached_get_selfseries_coeffs(s2; N)

    show() = "($s1 + $s2)"

    SymbolicSeries([], getNum, get_selfseries_coefficients, 
                   max(s1.contains_selfseries, s2.contains_selfseries), show, false)
end

Base.:+(s::SymbolicSeries, x::Number) = s + SymbolicSeries{0}(x)
Base.:+(x::Number, s::SymbolicSeries) = SymbolicSeries{0}(x) + s

function Base.:-(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 - new_s2
    end

    getNum(I::Vararg{Int, D}; N=nothing) = cached_getNum(s1, I...; N) - cached_getNum(s2, I...; N)

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        cached_get_selfseries_coeffs(s1, I...; N) ∪ cached_get_selfseries_coeffs(s2, I...; N)

    show() = "($s1 - $s2)"

    SymbolicSeries(s1.center, getNum, get_selfseries_coefficients, 
                   max(s1.contains_selfseries, s2.contains_selfseries), show, false)
end

function Base.:-(s::SymbolicSeries{D}, c::SymbolicSeries{0}) where D

    function getNum(I::Vararg{Int, D}; N=nothing)
        all(==(0), I) ? cached_getNum(s, I...; N) - cached_getNum(c; N) : cached_getNum(s, I...; N)
    end

    function get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing)
        if all(==(0), I) && isequal(cached_getNum(c; N), cached_getNum(s, I...; N))
            Set()
        else
            cached_get_selfseries_coeffs(s, I...; N) ∪ cached_get_selfseries_coeffs(c, N)
        end
    end

    show() = "($s - $c)"

    SymbolicSeries(s.center, getNum, get_selfseries_coefficients, 
                   max(s.contains_selfseries, c.contains_selfseries), show, false)
end
Base.:-(c::SymbolicSeries{0}, s::SymbolicSeries) = s-c

function Base.:-(s1::SymbolicSeries{0}, s2::SymbolicSeries{0})

    function getNum(;N=nothing)
        cached_getNum(s1 ;N) - cached_getNum(s2 ;N)
    end

    get_selfseries_coefficients(;N=nothing) = cached_get_selfseries_coeffs(s1; N) ∪ cached_get_selfseries_coeffs(s2; N)

    show() = "($s1 - $s2)"

    SymbolicSeries([], getNum, get_selfseries_coefficients, 
                   max(s1.contains_selfseries, s2.contains_selfseries), show, false)
end

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
    getNum(I::Int...; N=nothing) = t*cached_getNum(s, I...; N)
    show() = "($t * $s)"
    SymbolicSeries(s.center, getNum, s.get_selfseries_coefficients, s.contains_selfseries, show, false)
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
    getNum(I::Int...; N=nothing) = cached_getNum(s, I...; N) / t
    show() = "($s / $t)"
    SymbolicSeries(s.center, getNum, s.get_selfseries_coefficients, s.contains_selfseries, show, false)
end

function Base.:*(s1::SymbolicSeries{D}, s2::SymbolicSeries{D}) where D
    
    if !isequal(s1.center, s2.center)
        new_s1, new_s2 = merge_centers(s1, s2)
        return new_s1 * new_s2
    end

    function getNum(I::Vararg{Int, D}; N=nothing)
        res = 0
        for idx in CartesianIndices(tuple([0:i for i in I]...))
            res += cached_getNum(s1, Tuple(idx)...; N)*cached_getNum(s2, (I .- Tuple(idx))...; N)
        end
        res
    end

    function get_self_series_coefficients(I::Vararg{Int, D}; N=nothing)
        ranges = CartesianIndices(tuple([0:i for i in I]...))
        res = Set()
        for idx in ranges
            if !isequal(cached_getNum(s1, Tuple(idx)...; N), 0)
                res = res ∪ cached_get_selfseries_coeffs(s2, (I.-Tuple(idx))...; N=N)
            end
            if !isequal(cached_getNum(s2, (I.-Tuple(idx))...; N), 0)
                res = res ∪ cached_get_selfseries_coeffs(s1, Tuple(idx)...; N=N)
            end
        end
        res
    end

    show() = "($s1 * $s2)"

    SymbolicSeries(s1.center, getNum, get_self_series_coefficients, 
                   max(s1.contains_selfseries, s2.contains_selfseries), show, true)

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

    getNum(I::Vararg{Int, D}; N=nothing) = cached_getNum(s, I[perm]...; N)

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = 
        cached_get_selfseries_coeffs(s.series, I[perm]...; N)

    SymbolicSeries(new_center, getNum, get_selfseries_coefficients, s.series.contains_selfseries, s.series.show, false)(new_vars...)
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

    getNum(I::Vararg{Int}; N=nothing) = all(==(0), I[D1+1:D]) ? cached_getNum(s, I[1:D1]...; N) : 0

    get_selfseries_coefficients(I::Vararg{Int}; N=nothing) = all(==(0), I[D1+1:D]) ? 
        cached_get_selfseries_coeffs(s.series, I[1:D1]...; N=N) : Set()

    SymbolicSeries(new_center, getNum, get_selfseries_coefficients, s.series.contains_selfseries, s.series.show, false)(new_vars...)
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
Base.:+(s1::EvaluatedSymbolicSeries{0}, s2::EvaluatedSymbolicSeries{0}) = (s1.series+s2.series)()

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
Base.:-(s1::EvaluatedSymbolicSeries{0}, s2::EvaluatedSymbolicSeries{0}) = (s1.series-s2.series)()

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

    # handle differentiation at order greater than 1
    new_s = d.order == 1 ? s : Differential(d.x, d.order-1)(s)
    
    getNum(I::Vararg{Int, D}; N=nothing) = (I[x_idx] + 1)*cached_getNum(new_s.series, 
                                                                        I[1:(x_idx-1)]..., I[x_idx]+1, I[(x_idx+1):D]...; 
                                                                        N=isnothing(N) ? nothing : N+1)

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = cached_get_selfseries_coeffs(new_s.series, I[1:(x_idx-1)]..., I[x_idx]+1, I[x_idx+1:end]...; (N = isnothing(N) ? N : N+1))
    
    show() = "∂$(d.x)($s)"

    EvaluatedSymbolicSeries(SymbolicSeries(new_s.series.center, getNum, get_selfseries_coefficients, 
                                           s.series.contains_selfseries ≥ 1 ? 2 : 0, show, false), 
                            new_s.variables)

                    


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

    function getNum(I::Vararg{Int, D}; N=nothing) 

        I[x_idx] == 0 ? 0 : cached_getNum(s.series, 
                                          I[1:(x_idx-1)]..., I[x_idx]-1, I[(x_idx+1):D]...; 
                                          N=isnothing(N) ? nothing : N-1) / I[x_idx]
    end

    get_selfseries_coefficients(I::Vararg{Int, D}; N=nothing) = cached_get_selfseries_coeffs(s.series, I[1:(x_idx-1)]..., I[x_idx]+1, I[x_idx+1:end]...; (N = isnothing(N) ? N : N-1))
    
    show() = "∫ ($s) d$x"

    EvaluatedSymbolicSeries(SymbolicSeries(s.series.center, getNum, get_selfseries_coefficients, 
                                           s.series.contains_selfseries, show, false), 
                            s.variables)

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

    Similar SymbolicSeries, this is a wrapper for operations on EvaluatedSymbolicSeries, 
    allowing the operation to not be expanded when writing the operation and instead, 
    to be expanded later when calling the `expand` function.

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

                    function getNum1(I::Int...; N=nothing)
                        getNum_aux(k::Int) = cached_getNum(s, I[1:other_x-1]..., I[other_x]-k, I[other_x+1:x_idx-1]..., k, I[x_idx:end]...; N)
                        res = 0
                        for k in 0:I[other_x]
                            res += getNum_aux(k)
                        end
                        res
                    end


                    function get_selfseries_coefficients1(idcs::Vararg{Int}; N=nothing)
                        res = Set()
                        for idx in 0:idcs[other_x]
                            res = res ∪ cached_get_selfseries_coeffs(s,
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

                    new_series = SymbolicSeries(center, getNum1, get_selfseries_coefficients1, s.contains_selfseries, s.show, true)
                    new_at = at[1:x_idx-1]..., at[x_idx+1:end]...
                    return evaluate(new_series, new_at...; _nbr_found=_nbr_found)
                
                else # the center components do not match => LocalizedPDESeries

                    new_center = copy(s.center)
                    new_center[x_idx] = s.center[other_x]
                    new_series = translate(s, new_center)

                    return evaluate(new_series, at...; _nbr_found=_nbr_found)
                    
                end

            end

        else # x is a constant

            if isequal(at[x_idx], s.center[x_idx]) # x matches center component => allow use of PDESeries
                
                getNum3(I::Int...; N=nothing) = cached_getNum(s, I[1:x_idx-1]..., 0, I[x_idx:end]...; N)

                get_selfseries_coefficients3(idcs::Vararg{Int}; N=nothing) = 
                    cached_get_selfseries_coeffs(s, idcs[1:x_idx-1]...,0,idcs[x_idx:end]...; N)

                center = copy(s.center)
                deleteat!(center, x_idx)

                new_series = SymbolicSeries(center, getNum3, get_selfseries_coefficients3, 
                                            s.contains_selfseries, s.show, false)
                new_at = at[1:x_idx-1]..., at[x_idx+1:end]...

                return evaluate(new_series, new_at...; _nbr_found=_nbr_found)

            else # x does not match center component => LocalizedPDESeries

                new_center = copy(s.center)
                new_center[x_idx] = at[x_idx]
                new_series = translate(s, new_center)

                return evaluate(new_series, at...; _nbr_found=_nbr_found)
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

Base.show(io::IO, eq::SymbolicSeriesEquation) = print(io, "$(eq.LHS) ~ $(eq.RHS)")

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
        res = res ∪ cached_get_selfseries_coeffs(eq.LHS, I..., N=N)
    end

    if eq.RHS.contains_selfseries ≥ 1
        res = res ∪ cached_get_selfseries_coeffs(eq.RHS, I..., N=N)
    end

    return res

end

############################### cached getNum function ##############################
"""
    getNum2(x, idx::Int...; N=nothing)

    Retrieves a numeric representation of something that represents a series at index idx,
    while applying truncation order N. 

    If no specific getNum2 function is defined for a given representation of a series, 
    x is returned. This can also be useful when one wants to apply getNum2 to a bunch of
    elements, some of which might be Numbers or something else.

    This function uses memoization to compute the numeric representations efficiently. 
    The memoized version of this function is `cached_getNum`. It has the same signature.
    This cache can be cleared using `clear_num_cache()`

    Currently implemented versions of getNum2 include:

    - `getNum2(x::Any, ::Vararg{Int}; N=nothing)` -- default
    - `getNum2(sss::ScalarSeriesSymbol, idx::Vararg{Int}; N=nothing)` -- returns the value
      of the series coefficients if it has already been computed, and its unique_sym if it
      refers to a series represented by a Symbol (such as :self)
    - `getNum2(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D`

    Some variants are also implemented:

    - `getSymbolics(sc::SeriesCoefficient)::Num` -- Retrieves the unique_sym of a 
      SeriesCoefficient
    - `getNum(sc::SeriesCoefficient)::Number` -- Retrieves the value of a SeriesCoefficient
      if it has already been computed or its unique_sym if the series it refers to is a 
      Symbol (such as :self). If the series it refers to is not a Symbol and the 
      coefficient has not yet been computed, throws an error

    `cached_getNum` is implemented with the following signatures:

    - `cached_getNum(x::Any, idx::Vararg{Int}; N=nothing)` -- defaults to getNum2
    - `cached_getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D` -- 
      stores result in an IdDict
    - `cached_getNum(s::EvaluatedSymbolicSeries, idx::Vararg{Int}; N=nothing)`
    - `cached_getNum(eq::SymbolicSeriesEquation, idx::Vararg{Int}; N=nothing)`
"""
getNum2(x::Any, ::Vararg{Int}; N=nothing) = x # default behaviour

getSymbolics(sc::SeriesCoefficient)::Num = sc.unique_sym

function getNum2(sc::SeriesCoefficient)::Number
    if !(sc.ps isa Symbol)
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

getNum2(sss::ScalarSeriesSymbol, idx::Vararg{Int}; N=nothing) = getNum2(sss[convertIndices_fullsym_to_trunc(idx...)...])

getNum2(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D = s.getNum(idx...; N)

const _num_cache = IdDict{Any, Dict{Any, Any}}()
cached_getNum(x::Any, idx::Vararg{Int}; N=nothing) = getNum2(x, idx...; N)
function cached_getNum(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D
    if s.should_cache
        d = get!(() -> Dict{Any, Any}(), _num_cache, s)
        get!(() -> getNum2(s, idx...; N), d, idx)
    else
        getNum2(s, idx...; N)
    end
end
cached_getNum(s::EvaluatedSymbolicSeries, idx::Vararg{Int}; N=nothing) = cached_getNum(s.series, idx...;N)

clear_num_cache() = empty!(_num_cache)

function cached_getNum(eq::SymbolicSeriesEquation, idx::Vararg{Int}; N=nothing)
    cached_getNum(eq.LHS, idx...; N) ~ cached_getNum(eq.RHS, idx...; N)
end

################## cached get_selfseries_coefficients function ###################

const _selfseries_cache = IdDict{Any, Dict{Any, Any}}()
clear_selfseries_cache() = empty(_selfseries_cache)

function cached_get_selfseries_coeffs(s::SymbolicSeries{D}, idx::Vararg{Int, D}; N=nothing) where D
    if s.contains_selfseries == 0
        return Set()
    elseif s.should_cache
        d = get!(() -> Dict{Any, Any}(), _selfseries_cache, s)
        get!(() -> s.get_selfseries_coefficients(idx...; N), d, idx)
    else
        s.get_selfseries_coefficients(idx...; N)
    end
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
            res isa UnexpandedEvaluatedSymbolicSeries ? cached_expand(res, N) : res
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
function compute_coefficients_aux!(ps::PDESeries{T}, N::Int; 
                                   solver, 
                                   verbose::Int=0) where T

    N ≤ ps.order && return
    
    N > ps.order+1 && compute_coefficients_aux!(ps, N-1; solver, verbose=verbose)
    

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

    clear_num_cache()
    
end

function compute_coefficients!(ps::PDESeries, N::Int; solver=QRFactorization, verbose::Int=0)
    compute_coefficients_aux!(ps, N; solver, verbose)
    clear_expand_cache()
    clear_selfseries_cache()
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
                               solver=QRFactorization, verbose=0, benchmark::Bool=false) where T

    # first generate all expected unknowns of order up to N
    unknowns = expected_unknowns_upto(ps.unknown, length(ps.variables), N) 
    nbr_coeffs = length(unknowns[1]) # will be used later
    unknowns = [unknowns...;]

    # expand UnexpandedSymbolicSeriesEquation if necessary
    ss_equations = map(eq -> cached_expand(eq, N), ps.equations)

    # then generate all equations of orders up to N
    eqs = Equation[]
    for ((i,eq), maxIntOrd) in zip(enumerate(ss_equations), ps.maxIntegrationOrders)
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

    clear_num_cache()
    clear_expand_cache()
    clear_selfseries_cache()

end