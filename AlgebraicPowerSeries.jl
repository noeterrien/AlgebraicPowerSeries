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
    - `indices_expr::Vector{Num}` -- The indices it refers to inside a scalar series. For 
      instance [i+2, j] in K₍ᵢ₊₂₎ⱼ where i and j are Symbolics variables
    - `indices::Vector{Num}` -- the indices that appear in the indices_expr
    - `index::NTuple{Int, D}` -- The index of the scalar series it refers to when using
      multidimensional series

    ### Examples

    - `SeriesCoefficient(ps::Union{AbstractPowerSeries{D}, Symbol}, 
                         sym::Num,
                         indices_expr::Vector
                         indices::Vector{Num},
                         index::NTuple{Int, D}) where D` -- default constructor

"""
struct SeriesCoefficient{D}
    ps::Union{AbstractPowerSeries{D}, Symbol}
    sym::Num
    unique_sym::Num
    indices_expr::Vector
    indices::Vector{Num}
    index::NTuple{D, Int}
end

function SeriesCoefficient(ps::Union{AbstractPowerSeries{D}, Symbol},
                           sym::Num,
                           indices_expr::Vector,
                           indices::Vector{Num},
                           index::NTuple{D,Int}) where D

    if ps == :self
        u_sym = Symbol(string(:self) * string(index) * string(Num.(indices_expr))) # unique id
    else
        u_sym = Symbol(string(ps.seriesID) * string(index) * string(Num.(indices_expr))) # unique id
    end
    u_sym, = @variables $u_sym

    SeriesCoefficient(ps, sym, u_sym, indices_expr, indices, index)
end

_exponent_parsing_dict = Dict('⁰' => '0', '¹' => '1', '²' => '2', '³' => '3', '⁴' => '4', 
                              '⁵' => '5', '⁶' => '6', '⁷' => '7', '⁸' => '8', '⁹' => '9')

"""
    parse_coeff_name(name::String)::Vector{String}

    Gets a SeriesCoefficient indices expressions that appear in a specifically formatted 
    expression

    ###Input

    - `name::String` -- The formatted expression of a series coefficient :

      * The character '_' indicates a new index in the coefficient scalar series indices
      * This character must then be followed by a sequence of characters s that can be
        parsed to a Num using Symbolics.unwrap(eval(Meta.parse(string(s))))

      For instance, K²³_(i+j)_j will output ["(i+j)","j"]

    ###Output
    
    A Vector{String} containing the coefficient indices expressions
"""
function parse_coeff_name(name::String)::Tuple{Vector{String}, Vector{String}}
    idcs = []
    
    found_first_index = false
    for c in name
        if c == '_'
            push!(idcs, "")
            found_first_index = true
        elseif found_first_index
            idcs[end] *= c
        end
    end

    idcs
end

"""
    getValue(sc::SeriesCoefficient, at::Vector{Int})

    Returns the value of the coefficient

    ###Input

    - `sc::SeriesCoefficient` -- a SeriesCoefficient
    - `at::Vector{Int}` -- The values the coefficient's indices should be replaced with

    ###Output

    If the SeriesCoefficient refers to a PowerSeries (and not :self), and the coefficient
    has already been computed, the value of this coefficient is returned.

    Otherwise, throws an error

"""
function getValue(sc::SeriesCoefficient, at::Vector{Int})
    if sc.ps!=:self
        # first evaluate the indices
        d = Dict([i=>v for (i,v) in zip(sc.indices, at)])
        idx = []
        for expr in sc.indices_expr
            push!(idx, Symbolics.value(substitute(expr, d, fold=Val(true))))
        end
        
        # then return the result
        return sc.ps.coefficients[sc.index...][convertIndices(idx...)]
    else
        throw(ArgumentError("Cannot return value of a coefficient which refers to series
                             :self"))
    end
end


"""
    getUniqueSym(sc::SeriesCoefficient)

    Returns unique_sym attribute of a SeriesCoefficient

    ###Input
    - `sc::SeriesCoefficient`

    ###Output
    sc.unique_sym
"""
getUniqueSym(sc::SeriesCoefficient) = sc.unique_sym

#-----------------------------------------------------------SeriesSymbol-------------------------------------------------------------

"""
    ScalarSeriesSymbol <: AbstractScalarSeriesSymbol

    A convenient way to define SeriesCoefficients

    ###Fields

    - `ps::Union{PowerSeries, Symbol, Nothing}` -- The PowerSeries it refers to or :self
    - `scalar_idx::Tuple` -- The index of the ScalarSeries in the PowerSeries matrix
    - `coefficients::Dict{Tuple, SeriesCoefficient}` -- The SeriesCoefficients stored

    ###Examples
    - `ScalarSeriesSymbol(ps::Union{PowerSeries, Symbol}, scalar_idx::Tuple, 
                          coefficients::vector{SeriesCoefficient})` -- default constructor
    - `ps[1,2]` -- (where ps is a PowerSeries) get a ScalarSeriesSymbol using the getindex
      method
"""
mutable struct ScalarSeriesSymbol <: AbstractScalarSeriesSymbol
    ps::Union{PowerSeries, Symbol, Nothing}
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
    Base.getindex(sss::ScalarSeriesSymbol, I::Vararg)

    Create (if necessary) and returns a SeriesCoefficient at index I. The coefficient is
    then saved in the ScalarSeriesSymbol to be accessible at any time
"""
function Base.getindex(sss::ScalarSeriesSymbol, I::Vararg)
    vI = Vector([I...])
    if vI ∈ keys(sss.coefficients)
        return sss.coefficients[vI]
    else
        if sss.ps == :self
            sym = Symbol(string(:self) * string(sss.scalar_idx) * string(Num.(vI)))
            sym, = @variables $sym
            sss.coefficients[vI] = SeriesCoefficient(:self, sym, vI, getAllVariables(vI), sss.scalar_idx)
        elseif sss.ps isa PowerSeries
            sym = Symbol(string(sss.ps.seriesID) * string(sss.scalar_idx) * string(Num.(vI)))
            sym, = @variables $sym
            sss.coefficients[vI] = SeriesCoefficient(sss.ps, sym, vI, getAllVariables(vI), sss.scalar_idx)
        else
            throw(ArgumentError("Cannot construct a SeriesCoefficient for a 
                                 ScalarSeriesSymbol which doesn't refer to any known 
                                 PowerSeries or :self"))
        end
        return sss.coefficients[vI]
    end
end

"""
    selfseries_symbols(K, size...)

    Generates the ScalarSeriesSymbol with reference to the series :self of size size.
    One can then easily generate SeriesCoefficients using K[scalar_series_idx][coefficient_indices]
"""
macro selfseries_symbols(K, size...)
    quote 
        ci = reshape(collect(CartesianIndices($size)), $size)
        $(esc(K)) = map(idx -> ScalarSeriesSymbol(:self, Tuple(idx), Dict()), ci)
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
        ps.coefficients[i] = []
        for same_order_coeffs in dvpt.coeffs
            for coeff in same_order_coeffs
                push!(ps.coefficients[i], coeff)
            end
        end
    end
    
    ps.order=N
end




#------------------------------------------------------------RecursiveSeries----------------------------------------------------
"""
    ExpandableFormula

    ### Fields

    - `efID::Symbol` -- A unique reference to the expandable formula which might be used
      to create unique symbols when expanding the formula
    - `sym::Num` -- A symbol that may represent the ExpandableFormula in a Symbolics
      relation
    - `unique_sym::Num` -- A unique symbol to represent the Expandable Formula in a 
      Symbolics relation
    - `formula::Num` -- The representation of the formula that must be expanded
    - `fixed_indices::Vector` -- Indices that are present but are not use for expansion.
      Might be either Num for abstract representation or Int for concrete representation
    - `varying_indices::Vector{Num}` -- Indices on which to expand
    - `varying_indices_ranges::Vector` -- The ranges between which varying indices should
      be expanded. Can depend on Int, fixed_indices or varying_indices. Ranges should be
      given as tuples (start, end) and order should correspond to the order of 
      varying_indices vector. Furthermore, the numeric range of the varying index should
      be deducible from the value of previous varying indices and fixed indices
    - `expandable_formulae::Vector{ExpandableFormula}` -- Other expandable formulae that
      may appear in the formula 
    - `series_coeffs::Vector{SeriesCoefficient}` -- The series coefficients that appear in
      the relation
    - `func` -- A function that takes a vector of expressions that are instances of formula
      in which the coefficients that depend on symbolical indices have been replaced by the
      corresponding numerical values, and returns the single expanded expression that 
      corresponds to the desired sym

    ### Notes

    Expandable formulae work in a tree like way, with expandable_formulae being the nodes
    and series coefficients being the leaves.

    ### Examples

    - `ExpandableFormula(efID::Symbol
                         sym::Num,
                         formula::Num,
                         fixed_indices::Vector,
                         varying_indices::Vector{Num},
                         varying_indices_ranges::Vector,
                         expandable_formulae::Vector{ExpandableFormula},
                         series_coeffs::Vector{SeriesCoefficient},
                         func)` -- default constructor


"""
struct ExpandableFormula
    efID::Symbol
    sym::Num
    unique_sym::Num
    formula::Num
    fixed_indices::Vector
    varying_indices::Vector{Num}
    varying_indices_ranges::Vector
    expandable_formulae::Vector{ExpandableFormula}
    series_coeffs::Vector{SeriesCoefficient}
    func
end

"""
    function make_unique_sym(formula, scs::Vector{SeriesCoefficient}, 
                             efe::Vector{ExpandableFormula})

    Replaces all symbols from the vectors scs and efe in the fomula by their unique symbolic
    representation

    ###Input

    - `formula` -- The formula in which symbols must be replaced
    - `scs::Vector{SeriesCoefficient}` -- All the series coefficient for which the symbolic
      representation should be changed
    - `efe::Vector{ExpandableFormula}` -- All the expandable formulae for which the symbolic
      representation should be changed

    ###Output

    Updated formula
"""
function make_unique_sym(formula, scs::Vector, efe::Vector)
    dsc = Dict([sc.sym=>sc.unique_sym for sc in scs])
    def = Dict([ef.sym=>ef.unique_sym for ef in efe])
    fsc = substitute(formula, dsc)
    substitute(fsc, def)
end

function ExpandableFormula(efID::Symbol,
                           sym::Num,
                           formula::Num,
                           fixed_indices::Vector,
                           varying_indices::Vector{Num},
                           varying_indices_ranges::Vector,
                           expandable_formulae::Vector,
                           series_coeffs::Vector,
                           func)

    
    u_sym = Symbol(string(efID) * string(fixed_indices)) # unique id
    u_sym, = @variables $u_sym

    ExpandableFormula(efID,
                      sym,
                      u_sym,
                      make_unique_sym(formula, series_coeffs, expandable_formulae), 
                      fixed_indices,
                      varying_indices,
                      varying_indices_ranges,
                      expandable_formulae,
                      series_coeffs,
                      func)
end

"""
    substitute_known(formula, coeffs::Vector{SeriesCoefficient})

    Substitute all coefficients from the Vector coeffs in the formula by their value and
    returns the unknown coefficients from the Vector

    ###Input

    - `formula` -- The formula in which to replace the coefficients
    - `coeffs::Vector{SeriesCoefficient}` -- The series coefficients to replace

    ###Output

    - `result` -- The resulting formula
    - `unknowns::Vector{SeriesCoefficient}` -- The remaining unknown coefficients
"""
function substitute_known(formula, coeffs::Vector{SeriesCoefficient})
    d = Dict()
    unknowns = []
    for coeff in coeffs
        if coeff.ps != :self
            indices = Int[] 
            for idx_expr in coeff.indices_expr
                push!(indices, 
                      Symbolics.value(idx_expr))
            end

            d[coeff.unique_sym] = getValue(coeff, indices)
        else
            push!(unknowns, coeff)
        end
    end
    return substitute(formula, d), unknowns
end

"""
    RecurrentRelation

    ### Fields

    - `relation::Equation` -- The representation of the relation
    - `indices::Vector{Num}` -- The indices that appear in the relation 
    - `indices_ranges::Vector{Tuple{Union{Num, Int, Symbol}, Union{Num, Int, Symbol}}}` -- 
      The range of the indices for which the relation is valid. The numerical range of one
      index must be deducible from the previous indices in the list. To denote +∞, use :∞ 
    - `series_coeffs::Vector{SeriesCoefficient}` -- The series coefficients that appear in 
      the relation
    - `expandable_formulae::Vector{ExpandableFormula}` -- The expandable formulae that may
      appear in the relation

    ### Examples

    - `RecurrentRelation(relation::Num,
                         indices::Vector{Num}
                         indices_ranges::Vector{Tuple{Union{Num, Int, Symbol}, 
                                                      Union{Num, Int, Symbol}}},
                         series_coeffs::Vector{SeriesCoefficient},
                         expandable_formulae::Vector{ExpandableFormula})` -- default constructor


"""
struct RecurrentRelation
    relation::Equation
    indices::Vector{Num}
    indices_ranges::Vector{Tuple{Union{Num, Int, Symbol}, Union{Num, Int, Symbol}}}
    series_coeffs::Vector{SeriesCoefficient}
    expandable_formulae::Vector{ExpandableFormula}

    function RecurrentRelation(relation::Equation,
                               indices::Vector{Num},
                               indices_ranges::Vector,
                               series_coeffs::Vector,
                               expandable_formulae::Vector) 
        new(make_unique_sym(relation, series_coeffs, expandable_formulae),
            indices,
            indices_ranges,
            series_coeffs,
            expandable_formulae
        )
    end
end





"""
    RecurrentSeries{T,D} <: PowerSeries{T,D}

    A concrete type representing a series defined by a **linear** relation of recurrence
    around a center c

    ### Fields

    - `seriesID::Symbol` -- A unique reference to the series that might be used to choose
      unique IDs for SeriesCoefficient
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `scalar_series_ref::Array{AbstractScalarSeriesSymbol, D}` -- An array of 
      AbstractScalarSeriesSymbol that is used to easily create and access SeriesCoefficient

    relations::Vector{RecurrentRelation} -- A number of relations that allows to compute
      the coefficients numerically


    ### Notes 

    ### Examples

    - `RecurrentSeries(seriesID::Symbol,
                         size::NTuple{D,Int},
                         variables::Vector{Num},
                         center::Vector,
                         relations::Vector{RecurrentRelation})` -- default constructor
"""
mutable struct RecurrentSeries{T,D} <: PowerSeries{T,D}

    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    scalar_series_ref::Array{ScalarSeriesSymbol, D}
    


    relations::Vector{RecurrentRelation}

    function RecurrentSeries{T,D}(seriesID::Symbol,
                         size::NTuple{D,Int},
                         variables::Vector{Num},
                         center::Vector,
                         coefficients::Array{Vector{T},D},
                         order::Int,
                         scalar_series_ref::Array{ScalarSeriesSymbol, D},
                         relations::Vector{RecurrentRelation}) where {T,D}
        rs = new(seriesID,
                 size,
                 variables,
                 center,
                 coefficients,
                 order,
                 scalar_series_ref,
                 relations)
        for sss in scalar_series_ref
            sss.ps = rs
        end
        return rs
    end
end

function RecurrentSeries{T}(seriesID::Symbol,
                            size::NTuple{D,Int},
                            variables::Vector{Num},
                            center::Vector,
                            relations::Vector{RecurrentRelation}) where {T,D}
    coefficients = Array{Vector{T},D}(undef, size)
    for i in eachindex(coefficients)
        coefficients[i] = []
    end
    scalar_series_ref = map(idx -> ScalarSeriesSymbol(nothing, idx, Dict()), keys(coefficients))
    RecurrentSeries{T,D}(seriesID, size, variables, center, coefficients, -1, 
                         scalar_series_ref, relations)
end

"""
    function substitute_coeff_indices(c::SeriesCoefficient, subst::Dict,
        N::Union{Int,Nothing}=nothing, ps::Union{RecurrentSeries, Nothing}=nothing)

    Given a dictionary that associates indices of the coefficient c to some other value,
    substitutes in c indices_expr and indices and returns the new corresponding 
    SeriesCoefficient

    ###Input

    - `c::SeriesCoefficient` -- The SeriesCoefficient the substitution should be applied to
    - `subst::Dict` -- The substitution dictionnary
    - `N::Union{Int,Nothing}=nothing` -- If provided as well as ps, if the first index of
      the series coefficient is numerical and strictly less than N, then the returned 
      coefficient should relate to series ps instead of :self
    - `ps::Union{RecurrentSeries, Nothing}=nothing`

    ###Output

    The new SeriesCoefficient obtained
"""
function substitute_coeff_indices(c::SeriesCoefficient, subst::Dict, 
        N::Union{Int,Nothing}=nothing, ps::Union{RecurrentSeries, Nothing}=nothing)
    
    new_idc_expr = Num[]
    for idx_expr in c.indices_expr
        push!(new_idc_expr, substitute(idx_expr, subst))
    end

    new_idc = Num[]
    for idx in c.indices
        push!(new_idc, substitute(idx, subst))
    end

    if (!isnothing(N) && !isnothing(ps) 
        && (c.ps == :self)
        && isempty(Symbolics.get_variables(new_idc_expr[1])) 
        && (new_idc_expr[1] < N)
       )
        SeriesCoefficient(ps, c.unique_sym, new_idc_expr, new_idc, c.index)
    else
        SeriesCoefficient(c.ps, c.unique_sym, new_idc_expr, new_idc, c.index)
    end

end

"""
    substitute_fixed_indices_in_ef(ef::ExpandableFormula, fixed_indices_values::Dict,
                                   N::Int, ps::RecurrentSeries)

    Replaces all occurences of the given fixed indices in ef and return the new 
    corresponding ExpandableFormula

    #Input

    - `ef::ExpandableFormula` -- The ExpandableFormula to which the substitution must be
      applied
    - `fixed_indices_values::Dict` -- The index=>value substitution dictionary
    - `N::Int` -- The order to which the RecurrentSeries is being expanded
    - `ps::RecurrentSeries` -- The RecurrentSeries that is being expanded

    #Output

    An ExpandableFormula with the subtituted indices

"""
function substitute_fixed_indices_in_ef(ef::ExpandableFormula, fixed_indices_values::Dict,
                                        N::Int, ps::RecurrentSeries)
    # efID::Symbol => same                                                           ✓
    # sym::Num, => same                                                              ✓
    # formula::Num, => substituted (indices, coeffs and efe)                         ✓
    # fixed_indices::Vector, => substituted (indices)                                ✓
    # varying_indices::Vector{Num},                                                  ✓
    # varying_indices_ranges::Vector, => substituted (indices)                       ✓
    # expandable_formulae::Vector{ExpandableFormula}, => substituted (indices, rec)  ✓
    # series_coeffs::Vector{SeriesCoefficient}, => substituted (indices)             ✓
    # func => same                                                                   ✓

    new_fixed_indices = substitute_in_vector(ef.fixed_indices, fixed_indices_values)
    new_varying_indices_ranges = substitute_in_pairs_vector(ef.varying_indices_ranges, fixed_indices_values)
    new_series_coeffs = [substitute_coeff_indices(c, fixed_indices_values, N, ps) for c in ef.series_coeffs]
    new_expandable_formulae = [substitute_fixed_indices_in_ef(_ef, fixed_indices_values, N, ps) for _ef in ef.expandable_formulae]

    subst_dict = copy(fixed_indices_values)
    for (new_c, c) in zip(new_series_coeffs, ef.series_coeffs)
        subst_dict[c.unique_sym] = new_c.unique_sym
    end
    for (new_ef, ef) in zip(new_expandable_formulae, ef.expandable_formulae)
        subst_dict[ef.unique_sym] = new_ef.unique_sym
    end

    new_formula = substitute(ef.formula, subst_dict)

    ExpandableFormula(ef.efID, ef.sym, new_formula, new_fixed_indices, ef.varying_indices, 
                      new_varying_indices_ranges, new_expandable_formulae, 
                      new_series_coeffs, ef.func)
end

"""
    expand(ef::ExpandableFormula, indices_values::Dict, N::Int, ps::RecurrentSeries)

    Expands an ExpandableFormula to a single formula (instance of Num)

    ###Input

    - `ef::ExpandableFormula` -- The formula to expand
    - `fixed_indices_values::Dict` -- Fixed values that must be attributed to fixed
      indices before expanding. All other indices values must be deducible from these ones
    - `N::Int` -- The order to which the RecurrentSeries is being expanded
    - `ps::RecurrentSeries` -- The RecurrentSeries from which this expandable formula is 
      issued

    ###Output

    An instance of Num representing the expanded formula, and a Vector of unknown variables
    that appear in it
"""
function expand(ef::ExpandableFormula, fixed_indices_values::Dict, N::Int, ps::RecurrentSeries)

    if isempty(ef.varying_indices) # all indices values are stored in fixed_indices_values
        
        new_ef = substitute_fixed_indices_in_ef(ef, fixed_indices_values, N, ps)
        res_formula, res_unknowns = substitute_known(new_ef.formula, new_ef.series_coeffs)
        
        if isempty(new_ef.expandable_formulae)
            
            return res_formula, getUniqueSym.(res_unknowns)
        
        else

            fully_expanded_formulae, fully_expanded_unknowns = Num[], getUniqueSym.(res_unknowns)
            for to_expand in new_ef.expandable_formulae
                to_expand_formula, to_expand_unknowns = expand(to_expand, Dict(), N, ps)
                expanded_formula = substitute(res_formula, Dict(to_expand.unique_sym=>
                                                                      to_expand_formula))
                push!(fully_expanded_formulae, expanded_formula)
                fully_expanded_unknowns = [fully_expanded_unknowns;to_expand_unknowns]
            end
            fully_expanded_formula = ef.func(fully_expanded_formulae)

            return fully_expanded_formula, fully_expanded_unknowns

        end

    else # varying indices must be treated

        # compute the range of the first varying index
        a = Symbolics.value(substitute(ef.varying_indices_ranges[1][1], fixed_indices_values))
        b = Symbolics.value(substitute(ef.varying_indices_ranges[1][2], fixed_indices_values))

        # compute all the sub-ExpandableFormulae
        fully_expanded_formulae, fully_expanded_unknowns = Num[], Num[]
        for i in a:b 
            new_fixed_indices = copy(ef.fixed_indices)
            new_varying_indices = ef.varying_indices[2:end]
            new_varying_indices_ranges = ef.varying_indices_ranges[2:end]
            new_fixed_indices_values = copy(fixed_indices_values)
            new_fixed_indices_values[ef.varying_indices[1]] = i

            to_expand = ExpandableFormula(ef.efID, ef.sym, ef.formula, new_fixed_indices,
                                          new_varying_indices, new_varying_indices_ranges, 
                                          ef.expandable_formulae, ef.series_coeffs, ef.func)

            to_expand_formula, to_expand_unknowns = expand(to_expand, new_fixed_indices_values, N, ps)
            push!(fully_expanded_formulae, to_expand_formula)
            fully_expanded_unknowns = [fully_expanded_unknowns; to_expand_unknowns]
        end

        return ef.func(fully_expanded_formulae), fully_expanded_unknowns

    end

end

"""
    function iterate_expand(truncated_rr::RecurrentRelation, N::Int, ps::RecurrentSeries, 
                            fixed_values::Dict=Dict(), k::Int=1)

    Expands a recurrent relations over its indices

    ###Input

    - `truncated_rr::RecurrentRelation` -- A recurrent relation with **finite** ranges
    - `N::Int` -- order at which the relation is currently being expanded. Is used to
      replace coefficients of lesser order with their values
    - `ps::RecurrentSeries` -- The RecurrentSeries from which the relation is being 
      expanded
    - `fixed_values::Dict=Dict()` -- A dictionary indicating what are the fixed indices
      inside the RecurrentRelation and what are their values
    - `k::Int=1` -- Internal variables used for recursion


    ###Output

    - `equations::Vector{Equation}` -- see expand(rr::RecurrentRelation, N::Int)
    - `unknowns::Vector{Num}` -- see expand(rr::RecurrentRelation, N::Int)
"""
function iterate_expand(rr::RecurrentRelation, N::Int, 
                        ps::RecurrentSeries, fixed_values::Dict=Dict(), k::Int=1)

    if isempty(rr.indices_ranges) # each index is determined by the fixed_values dict
        

        # Substitute indices
        subst_dict = copy(fixed_values)

        # Substitute coefficients
        unknowns = SeriesCoefficient[]
        for c in rr.series_coeffs
            new_sc = substitute_coeff_indices(c, fixed_values, N, ps)
            push!(unknowns, new_sc)
            subst_dict[c.unique_sym] = new_sc.unique_sym
        end
        res_relation = substitute(rr.relation, subst_dict)
        res_relation, unknowns = substitute_known(res_relation, unknowns)
        res_unknowns = getUniqueSym.(unknowns)

        # Substitute expandable formulae
        for ef in rr.expandable_formulae
            expanded_ef_formula, new_unknowns = expand(ef, fixed_values, N, ps)
            res_unknowns = [res_unknowns; new_unknowns]
            subst_dict[ef.unique_sym] = expanded_ef_formula
        end

        res_relation = substitute(res_relation, subst_dict)

        res_unknowns
        return [res_relation], res_unknowns

    else

        all_equations, all_unknowns = Equation[], Num[]
        rg = rr.indices_ranges[1] # first range should be deducible from fixed_values
        for i in Symbolics.value(substitute(rg[1], fixed_values)):Symbolics.value(
                 substitute(rg[2], fixed_values))
            new_indices_ranges = rr.indices_ranges[2:end]
            new_fixed_values = copy(fixed_values)
            new_fixed_values[rr.indices[k]] = i

            new_rr = RecurrentRelation(rr.relation, rr.indices, new_indices_ranges, 
                                       rr.series_coeffs, rr.expandable_formulae)
            
            equations, unknowns = iterate_expand(new_rr, N, ps, new_fixed_values, k+1)
            all_equations = [all_equations;equations]
            all_unknowns = [all_unknowns;unknowns]
        end

        return all_equations, all_unknowns

    end

end

"""
    function expand(rr::RecurrentRelation, N::int, ps::RecurrentSeries)
    
    Expands the recurrent relation for all possible indices (:∞ is replaced by N)

    ###Input
    
    - `rr::RecurrentRelation` -- A relation of recurrence
    - `N::Int` -- The order up to which the relation should be expanded
    - `ps::RecurrentSeries` -- The series from which the relation is issued 

    ###Output

    - `equations::Vector{Equation}` -- A vector of equations
    - `unknowns::Vector{Num}` -- The unknown variables that appear in these equations 

"""
function expand(rr::RecurrentRelation, N::Int, ps::RecurrentSeries)

    new_indices_ranges = NTuple{2, Union{Int, Num}}[]
    for index_range in rr.indices_ranges
        if index_range[2] == :∞
            push!(new_indices_ranges, (max(N, index_range[1]),N))
        else
            push!(new_indices_ranges, index_range)
        end
    end


    # Truncate
    truncated_rr = RecurrentRelation(rr.relation,
                                     rr.indices,
                                     new_indices_ranges,
                                     rr.series_coeffs,
                                     rr.expandable_formulae
    )

    # Expand the truncated relation
    iterate_expand(truncated_rr, N, ps)
end

"""
    compute_coefficients!(ps::RecurrentSeries, N::Int)

    Computes the coefficients of a RecurrentSeries up to order N

    ###Input 
    
    - `ps::RecurrentSeries` -- a RecurrentSeries
    - `N::Int` -- The order up to which coefficients should be computed
    - `verbose=false` -- named argument. If set to true, indicates when a new order is 
      being computed and when it is done in the console

    ###Output

    Nothing
"""
function compute_coefficients!(ps::RecurrentSeries{T,D}, N::Int; verbose=false) where {T,D}

    if ps.order ≥ N 
        return
    end
    
    if ps.order < N-1
        compute_coefficients!(ps, N-1; verbose=verbose)
    end
    
    verbose && println("Computing coefficients of order $N")
    # compute for order N
    # Expand all equations
    equations, unknowns = Equation[], Num[]
    for rr in ps.relations
        new_equations, new_unknowns = expand(rr, N, ps)
        equations = [equations; new_equations]
        unknowns = [unknowns; new_unknowns]
    end
    unknowns = unique(unknowns)

    # solve for coefficients
    unsorted_coeffs = symbolic_linear_solve(equations, unknowns)
    unsorted_coeffs = Symbolics.value.(unsorted_coeffs)

    # set series coefficients
    unknowns_idx = decode_coeffIndexAndIndices.(unknowns)
    ## find maximum series coefficient index
    max_idx = maximum(t -> t[2], unknowns_idx)
    ## set ps.coefficients to the correct dimensions
    for i in eachindex(ps.coefficients)
            old_length = length(ps.coefficients[i])
            new_spaces = Vector{T}(undef, max_idx-old_length)
            ps.coefficients[i] = [ps.coefficients[i];new_spaces]
    end
    ## fill
    for (idx, val) in zip(unknowns_idx, unsorted_coeffs) 
        ps.coefficients[idx[1]...][idx[2]] = val
    end
    ps.order += 1

    verbose && println("Coefficients computed up to order $N")
    return
end