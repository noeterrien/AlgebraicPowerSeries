using Symbolics
include("utilitaries.jl")

#-------------------------------------------------------------PowerSeries--------------------------------------------------------
abstract type AbstractPowerSeries{D} end

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

    - `compute_coefficients!(ps::PowerSeries, N::Int)` -- computes the coefficients up to 
      order N
"""
abstract type PowerSeries{T,D} <: AbstractPowerSeries{D} end  

"""
    Base.getindex(ps::PowerSeries{Dps}, I::Vararg{Int64,Didx}) where {Dps,Didx}

    Access the series coefficients

    ###Input

    - `ps::PowerSeries{Dps}` -- a PowerSeries
    - `I::Vararg{Int64,Didx}` -- the indices of the coefficient. Didx <= Dps+1. The first
      Dps indices allow the selection of serie(s) in the array of coupled series while the
      Dps+1 index allows selects the order of the coefficient (starting at 0)

    ###Output

    Depending on the number of indices, either an Array{Vector{T},Dps-Didx} if Didx < Dps,
    a Vector{T} if Didx == Dps or a T if Didx > Dps.
    Will throw an ArgumentError if trying to access a coefficient that has not been
    computed yet
"""
function Base.getindex(ps::PowerSeries{T,Dps}, I::Vararg{Int64,Didx}) where {T,Dps,Didx}
    if Didx <= Dps
        return ps.coefficients[I...]
    else 
        coeffs = ps.coefficients[I[1:Dps]...]
        idx = convertIndices(I[Dps+1:Didx]...)
        if idx <= length(coeffs)
            return coeffs[idx]
        else
            throw(ArgumentError("Trying to access a series coefficient that has not been 
                                 computed yet"))
        end
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
    
    monomials = compute_monomials(N, ps.variables)
    to_build = zeros(Num, ps.size)
    for i in eachindex(ps.coefficients)
        to_build[i] = sum(ps[i][1:length(monomials)] .* monomials)
    end

    map(x -> build_function(x, ps.variables; expression=Val{false}), to_build)
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
        T = typeof(built[1](args))
        res = Array{T}(undef, ps.size)
        for i in eachindex(res)
            res[i] = built[i](args)
        end
        res
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
        return sc.ps[sc.index..., idx...]
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

#-----------------------------------------------------------TaylorSeries-------------------------------------------------------------


"""
    TaylorSeries{T,D} <: PowerSeries{T,D}

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
    - `last_computed_derivatives::Vector{Num}` -- Symbolics representation of the last 
      derivatives of func that were computed to compute the coefficients
    - `origin_eval_dict::Dict{Num}` -- A dict to associate each symbol to its value at 
      the center of the series
    - `differentials::Vector{Differential}` -- A vector to store the differential operators
      with respect to the different variables
    - `factorials::Vector{Int}` -- A vector to store the factorial coefficients to apply
      to each differential when computing coefficients
    - `factorials_orders::Vector{Vector{Int}}` -- A vector to store the corresponding orders of
      each variable in the factorials vector
    - `dp::Vector{Int}` -- A vector to store the position differentials computation should be
      started at when computing coefficients

    ### Notes
    
    Tested types T include Float64 and ComplexF64. T is infered 

    ### Examples

    - `TaylorSeries{T}(seriesID::Symbol,
                       variables::Vector{Num}, 
                       func::Array{Num, D},
                       center::Vector)` -- default constructor
"""
mutable struct TaylorSeries{T,D} <: PowerSeries{T,D}
    
    seriesID::Symbol
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    func::Array{Num, D}

    # used for coefficients computation
    last_computed_derivatives::Vector{Array{Num,D}}
    origin_eval_dict::Dict{Num}
    differentials::Vector{Differential}
    factorials::Vector{Int}
    factorials_orders::Vector{Vector{Int}}
    dp::Vector{Int}
end

function TaylorSeries{T}(seriesID::Symbol,
                        variables::Vector{Num}, 
                        func::Array{Num, D}, 
                        center::Vector) where {T,D}
    if length(variables)==length(center)
        origin_eval_dict = Dict([v=>c for (v,c) in zip(variables, center)])
        differentials = [Differential(v) for v in variables]
        # create coefficients array
        coeffs = Array{Vector{T}}(undef, size(func))
        for i in eachindex(coeffs)
            coeffs[i] = Vector{T}()
        end

        TaylorSeries(seriesID, size(func), variables, center, coeffs, -1, func, 
                     Array{Num,D}[], origin_eval_dict, 
                     differentials, Int[], Vector{Int}[], Int[])
    else
        throw(ArgumentError("center size does not match number of variables"))
    end
end

"""
    compute_coefficients!(ps::TaylorSeries, N::Int)

    Computes the coefficients of a TaylorSeries up to order N

    ###Input 
    
    - `ps::TaylorSeries` -- a TaylorSeries
    - `N::Int` -- The order up to which coefficients should be computed

    ###Output

    Nothing
"""
function compute_coefficients!(ps::TaylorSeries{T}, N::Int) where T
    # first check to which order coefficients have already been computed
    if ps.order >= N
        return
    else
        # compute up to order N
        if ps.order < N-1
            compute_coefficients!(ps, N-1)
        end

        # compute derivatives and factorials
        last_computed_derivatives = copy(ps.last_computed_derivatives)
        factorials = copy(ps.factorials)
        factorials_orders = copy(ps.factorials_orders)
        dp = copy(ps.dp)
        if isempty(last_computed_derivatives)
            ps.last_computed_derivatives = [ps.func]
            ps.factorials = [1]
            ps.factorials_orders = [zeros(Int, length(ps.variables))]
            ps.dp = ones(length(ps.variables))
        else 
            ps.last_computed_derivatives = []
            ps.factorials = []
            ps.factorials_orders = Vector{Int}[]
            new_dp = 1
            for (vidx,diff) in enumerate(ps.differentials)
                ps.dp[vidx] = new_dp
                for (d,f,fo) in zip(last_computed_derivatives[dp[vidx]:end], 
                                    factorials[dp[vidx]:end], 
                                    factorials_orders[dp[vidx]:end])
                    push!(ps.last_computed_derivatives, expand_derivatives.(diff.(d)))
                    push!(ps.factorials, f*(fo[vidx]+1))
                    new_order = copy(fo)
                    new_order[vidx] = new_order[vidx] + 1
                    push!(ps.factorials_orders,new_order)
                end
                nbr_computed = length(factorials)-(dp[vidx]-1)
                new_dp += nbr_computed
            end
        end
        
        # compute coefficients of order N
        for (func,f) in zip(ps.last_computed_derivatives, ps.factorials)
            coeff = substitute(func, ps.origin_eval_dict, fold=Val(true))./f
            for i in eachindex(ps.coefficients)
                val = Symbolics.value(coeff[i])
                push!(ps.coefficients[i], (val |> T))
            end
        end

        ps.order=N

    end
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
    function substitute_coeff_indices(c::SeriesCoefficient, subst::Dict)

    Given a dictionary that associates indices of the coefficient c to some other value,
    substitutes in c indices_expr and indices and returns the new corresponding 
    SeriesCoefficient

    ###Input

    - `c::SeriesCoefficient` -- The SeriesCoefficient the substitution should be applied to
    - `subst::Dict` -- The substitution dictionnary

    ###Output

    The new SeriesCoefficient obtained
"""
function substitute_coeff_indices(c::SeriesCoefficient, subst::Dict)
    
    new_idc_expr = Num[]
    for idx_expr in c.indices_expr
        push!(new_idc_expr, substitute(idx_expr, subst))
    end

    new_idc = Num[]
    for idx in c.indices
        push!(new_idc, substitute(idx, subst))
    end

    SeriesCoefficient(c.ps, c.unique_sym, new_idc_expr, new_idc, c.index)

end

"""
    substitute_fixed_indices_in_ef(ef::ExpandableFormula, fixed_indices_values::Dict)

    Replaces all occurences of the given fixed indices in ef and return the new 
    corresponding ExpandableFormula

    #Input

    - `ef::ExpandableFormula` -- The ExpandableFormula to which the substitution must be
      applied
    - `fixed_indices_values::Dict` -- The index=>value substitution dictionary

    #Output

    An ExpandableFormula with the subtituted indices

"""
function substitute_fixed_indices_in_ef(ef::ExpandableFormula, fixed_indices_values::Dict)
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
    new_series_coeffs = [substitute_coeff_indices(c, fixed_indices_values) for c in ef.series_coeffs]
    new_expandable_formulae = [substitute_fixed_indices_in_ef(_ef, fixed_indices_values) for _ef in ef.expandable_formulae]

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
    expand(ef::ExpandableFormula, indices_values::Dict)

    Expands an ExpandableFormula to a single formula (instance of Num)

    ###Input

    - `ef::ExpandableFormula` -- The formula to expand
    - `fixed_indices_values::Dict` -- Fixed values that must be attributed to fixed
      indices before expanding. All other indices values must be deducible from these ones


    ###Output

    An instance of Num representing the expanded formula, and a Vector of unknown variables
    that appear in it
"""
function expand(ef::ExpandableFormula, fixed_indices_values::Dict)

    if isempty(ef.varying_indices) # all indices values are stored in fixed_indices_values
        
        new_ef = substitute_fixed_indices_in_ef(ef, fixed_indices_values)
        res_formula, res_unknowns = substitute_known(new_ef.formula, new_ef.series_coeffs)
        
        if isempty(new_ef.expandable_formulae)
            
            return res_formula, getUniqueSym.(res_unknowns)
        
        else

            fully_expanded_formulae, fully_expanded_unknowns = Num[], getUniqueSym.(res_unknowns)
            for to_expand in new_ef.expandable_formulae
                to_expand_formula, to_expand_unknowns = expand(to_expand, Dict())
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

            to_expand_formula, to_expand_unknowns = expand(to_expand, new_fixed_indices_values)
            push!(fully_expanded_formulae, to_expand_formula)
            fully_expanded_unknowns = [fully_expanded_unknowns; to_expand_unknowns]
        end

        return ef.func(fully_expanded_formulae), fully_expanded_unknowns

    end

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
    function iterate_expand(truncated_rr::RecurrentRelation, fixed_values::Dict=Dict())

    Expands a recurrent relations over its indices

    ###Input

    - `truncated_rr::RecurrentRelation` -- A recurrent relation with **finite** ranges
    - `fixed_values::Dict` -- A dictionary indicating what are the fixed indices
      inside the RecurrentRelation and what are their values


    ###Output

    - `equations::Vector{Equation}` -- see expand(rr::RecurrentRelation, N::Int)
    - `unknowns::Vector{Num}` -- see expand(rr::RecurrentRelation, N::Int)
"""
function iterate_expand(rr::RecurrentRelation, fixed_values::Dict=Dict(), k::Int=1)

    if isempty(rr.indices_ranges) # each index is determined by the fixed_values dict
        

        # Substitute indices
        subst_dict = copy(fixed_values)

        # Substitute coefficients
        unknowns = SeriesCoefficient[]
        for c in rr.series_coeffs
            new_sc = substitute_coeff_indices(c, fixed_values)
            push!(unknowns, new_sc)
            subst_dict[c.unique_sym] = new_sc.unique_sym
        end
        res_relation = substitute(rr.relation, subst_dict)
        res_relation, unknowns = substitute_known(res_relation, unknowns)
        res_unknowns = getUniqueSym.(unknowns)

        # Substitute expandable formulae
        for ef in rr.expandable_formulae
            expanded_ef_formula, new_unknowns = expand(ef, fixed_values)
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
            
            equations, unknowns = iterate_expand(new_rr, new_fixed_values, k+1)
            all_equations = [all_equations;equations]
            all_unknowns = [all_unknowns;unknowns]
        end

        return all_equations, all_unknowns

    end

end

"""
    function expand(rr::RecurrentRelation, N::int)
    
    Expands the recurrent relation for all possible indices (:∞ is replaced by N)

    ###Input
    
    - `rr::RecurrentRelation` -- A relation of recurrence
    - `N::Int` -- The order up to which the relation should be expanded

    ###Output

    - `equations::Vector{Equation}` -- A vector of equations
    - `unknowns::Vector{Num}` -- The unknown variables that appear in these equations 

"""
function expand(rr::RecurrentRelation, N::Int)

    # Truncate
    truncated_rr = RecurrentRelation(rr.relation,
                                     rr.indices,
                                     substitute_in_pairs_vector(rr.indices_ranges, Dict(:∞ => N)),
                                     rr.series_coeffs,
                                     rr.expandable_formulae
    )

    # Expand the truncated relation
    iterate_expand(truncated_rr)
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

    relations::Vector{RecurrentRelation}

end

function RecurrentSeries{T}(seriesID::Symbol,
                            size::NTuple{D,Int},
                            variables::Vector{Num},
                            center::Vector,
                            relations::Vector{RecurrentRelation}) where {T,D}
    RecurrentSeries{T,D}(seriesID, size, variables, center, 
                         Array{Vector{T},D}(undef, size), -1, relations)
end


"""
    compute_coefficients!(ps::RecurrentSeries, N::Int)

    Computes the coefficients of a RecurrentSeries up to order N

    ###Input 
    
    - `ps::RecurrentSeries` -- a RecurrentSeries
    - `N::Int` -- The order up to which coefficients should be computed

    ###Output

    Nothing
"""
function compute_coefficients!(ps::RecurrentSeries{T,D}, N::Int) where {T,D}

    # Expand all equations
    equations, unknowns = Equation[], Num[]
    for rr in ps.relations
        new_equations, new_unknowns = expand(rr, N)
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
        ps.coefficients[i] = Vector{T}(undef, max_idx+1)
    end
    ## fill
    for (idx, val) in zip(unknowns_idx, unsorted_coeffs) 
        ps.coefficients[idx[1]...][idx[2]] = val
    end
    ps.order = N
    
end