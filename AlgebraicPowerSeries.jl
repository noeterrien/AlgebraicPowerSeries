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

    - `compute_coefficients(ps::PowerSeries, N::Int)` -- computes the coefficients up to 
      order N
"""
abstract type PowerSeries{T,D} <: AbstractPowerSeries{D} end  

"""
    convertIndices(I::Vararg{Int64, Didx}) where Didx

    Converts indices of the form a₀₀₀ + a₁₀₀ x + a₁₁₀ y + a₁₁₁ z + a₂₀₀ x² + a₂₁₀ xy + 
    a₂₁₁ xz + a₂₂₀ y² + a₂₂₁ yz + a₂₂₂ z² + ... to an index of the the form 
    b₁ + b₂ x + b₃ y + b₄ z + b₅ x² + b₆ xy + b₇ xz + b₈ y² + b₉ yz + b₁₀ z² + ... 
"""
function convertIndices(I::Vararg{Int64, Didx}) where Didx
    idx = 1
    for (n,i) in zip(Didx:-1:1, I)
        idx += Base.sum([binomial(k+n-1,n-1) for k in 0:i-1])
    end
    idx
end

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
                         indices_expr::Vector{Num}
                         indices::Vector{Num},
                         index::NTuple{Int, D}) where D` -- default constructor

"""
struct SeriesCoefficient{D}
    ps::Union{AbstractPowerSeries{D}, Symbol}
    sym::Num
    unique_sym::Num
    indices_expr::Vector{Num}
    indices::Vector{Num}
    index::NTuple{D, Int}
end

function SeriesCoefficient(ps::Union{AbstractPowerSeries{D}, Symbol},
                           sym::Num,
                           indices_expr::Vector{Num},
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
                       Array{Num, D},
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
    compute_coefficients(ps::TaylorSeries, N::Int)

    Computes the coefficients of a TaylorSeries up to order N

    ###Input 
    
    - `ps::TaylorSeries` -- a TaylorSeries
    - `N::Int` -- The order up to which coefficients should be computed

    ###Output

    Nothing
"""
function compute_coefficients(ps::TaylorSeries{T}, N::Int) where T
    # first check to which order coefficients have already been computed
    if ps.order >= N
        return
    else
        # compute up to order N
        if ps.order < N-1
            compute_coefficients(ps, N-1)
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
            indices = [] 
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
    expand(ef::ExpandableFormula, indices_values::Dict{Num, Int})

    expands an ExpandableFormula to a single formula (instance of Num)

    ###Input

    - `ef::ExpandableFormula` -- The formula to expand
    - `fixed_indices_values::Dict` -- Fixed values that must be attributed to fixed
      indices before expanding. All other indices values must be deducible from these ones


    ###Output

    An instance of Num representing the expanded formula, and a Vector of unknown variables
    that appear in it
"""
function expand(ef::ExpandableFormula, fixed_indices_values::Dict)

    
    subst_fixed = substitute_in_vector(ef.fixed_indices, fixed_indices_values)
    subst_var_rgs = substitute_in_pairs_vector(ef.varying_indices_ranges,
                                               fixed_indices_values)
    subst_formula_dict = Dict()
    subst_scs = []
    for sc in ef.series_coeffs
        fixed_sc = substitute_coeff_indices(sc, fixed_indices_values)
        subst_formula_dict[sc.unique_sym] = fixed_sc.unique_sym
        push!(subst_scs, fixed_sc)
    end
    @show ef.formula
    @show subst_formula_dict
    @show subst_formula = substitute(ef.formula, subst_formula_dict)
    subst_formula = substitute(subst_formula, fixed_indices_values)

    fixed_ef = ExpandableFormula(ef.efID, ef.sym, subst_formula,
                                 subst_fixed,
                                 ef.varying_indices,
                                 subst_var_rgs,
                                 ef.expandable_formulae,
                                 subst_scs,
                                 ef.func)

        
    
    if fixed_ef.varying_indices_ranges == [] # base case base case

        #substitute all known coefficients
        return substitute_known(fixed_ef.formula, fixed_ef.series_coeffs)


    elseif fixed_ef.expandable_formulae == [] # base case
        
        unknowns = [] # powerseries coefficients that have not been computed yet (bc cannot be)
        all_expanded = []
        for i in Symbolics.value(fixed_ef.varying_indices_ranges[1][1]):Symbolics.value(
                                 fixed_ef.varying_indices_ranges[1][2])
            
            # substitute in indices 
            new_fixed = copy(fixed_ef.fixed_indices)
            push!(new_fixed, fixed_ef.varying_indices[1])
            new_varying = fixed_ef.varying_indices[2:end]
            new_varying_ranges = substitute_in_pairs_vector(fixed_ef.varying_indices_ranges[2:end],
                                                            Dict(fixed_ef.varying_indices[1]=>i))
            new_fixed_values = copy(fixed_indices_values)
            new_fixed_values[fixed_ef.varying_indices[1]] = i


            # substitute in formula series coefficients
            new_scs = []
            subst_dict = Dict{Num, Num}()
            for coeff in ef.series_coeffs

                new_coeff = substitute_coeff_indices(coeff, Dict(ef.varying_indices[1]=>i))
                push!(new_scs, new_coeff)

                subst_dict[coeff.unique_sym] = new_coeff.unique_sym
            end

            # substitute in formula
            subst_dict[fixed_ef.varying_indices[1]] = i
            expanded_formula = substitute(fixed_ef.formula, subst_dict)
            expanded_ef = ExpandableFormula(fixed_ef.efID,
                                            fixed_ef.sym, 
                                            expanded_formula, 
                                            new_fixed,
                                            new_varying,
                                            new_varying_ranges,
                                            [],
                                            new_scs,
                                            fixed_ef.func)
            
            
            expanded, coeffs = expand(expanded_ef, new_fixed_values)
            unknowns = [unknowns;coeffs]
            push!(all_expanded, expanded)
        end

        return fixed_ef.func(all_expanded), unknowns

    else # need to expand other formulae first
        
        unknowns = [] # powerseries coefficients that have not been computed yet (bc cannot be)
        all_expanded = []
        for to_expand in ef.expandable_formulae
            expanded, coeffs = expand(to_expand, fixed_indices_values)
            push!(all_expanded, expanded)
            push!(unknowns, coeffs)
        end
        
        
        return fixed_ef.func(all_expanded), unknowns 

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
    function compute_ranges(ef::ExpandableFormula, ranges::Dict)
    
    Given an ExpandableFormula and an index=>value dictionnary, computes the new vector of
    ranges that should be given to expand to expand ef

    ###Input

    - `ef::ExpandableFormula` -- The ExpandableFormula one whishes to expand
    - `ranges::Dict` -- The index=>value dictionnary that one whishes to apply

    ###Output

    A Vector of ranges that can be given to expand function
"""
function compute_ranges(ef::ExpandableFormula, ranges::Dict)
    res = []
    for idx in ef.indices
        if haskey(ranges, idx)
            push!(res, (ranges[idx][1], ranges[idx][2]))
        end
    end
    res
end

"""
    function iterate_expand(truncated_rr::RecurrentRelation, it::Dict, temp::Dict)

    Expands a recurrent relations over multiple indices

    ###Input

    - `truncated_rr::RecurrentRelation` -- A recurrent relation with **finite** ranges
    - `it::Dict` -- An index to range dictionnary
    - `temp::Dict` -- temporary dictionnary containing the current attributed values of
      the different indices and the corresponding indices in truncated_rr.indices list

    ###Output

    - `equations::Vector{Equation}` -- see expand(rr::RecurrentRelation, N::Int)
    - `unknowns::Vector{Num}` -- see expand(rr::RecurrentRelation, N::Int)
"""
function iterate_expand(truncated_rr::RecurrentRelation, it::Dict, temp::Dict)

    unknowns = []

    if isempty(it) # base case

        # Replace indices that appear directly in the relation
        subst_dict = Dict([(k=>v) for (k,v) in temp]) #container for all substitutions
                                                         #that should be done at the end

        # Replace series coefficients by their value or expanded symbolic representation
        for c in truncated_rr.series_coeffs
            new_coeff = substitute_coeff_indices(c, subst_dict)
            subst_dict[c.unique_sym] = new_coeff.unique_sym
            push!(unknowns, new_coeff.unique_sym)
        end
        equation = substitute(truncated_rr.relation, subst_dict)
        equation, unknowns = substitute_known(equation, unknowns)

        # TODO : Replace expandable formulae by their expanded representation
        ef_subst_dict = Dict()
        for ef in truncated_rr.expandable_formulae

        end
        

        return [equation], unknowns

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

    # truncate
    truncated_rr = RecurrentRelation(rr.relation,
                                     rr.indices,
                                     substitute_in_pairs_vector(rr.indices_ranges, Dict(:∞ => N)),
                                     rr.expandable_formulae,
                                     rr.series_coeffs
    )

    # deduce indices over which it is possible to iterate (i.e range is defined by two Int)
    it = Dict()
    for (idx, rg) in zip(truncated_rr.indices, truncated_rr.indices_ranges)
        if (rg[1] isa Int) && (rg[2] isa Int) it[idx] = rg end
    end 

    iterate_expand(truncated_rr, it, Dict())
end





"""
    RecurrentSeries{T,D} <: PowerSeries{T,D}

    A concrete type representing a series defined by a relation of recurrence around a center c

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
struct RecurrentSeries{T,D} <: PowerSeries{T,D}

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
    RecurrentSeries{T,D}(seriesID, size, variables, center, [], -1, relations)
end


"""
    compute_coefficients(ps::RecurrentSeries, N::Int)

    Computes the coefficients of a RecurrentSeries up to order N

    ###Input 
    
    - `ps::RecurrentSeries` -- a RecurrentSeries
    - `N::Int` -- The order up to which coefficients should be computed

    ###Output

    Nothing
"""
function compute_coefficients(ps::RecurrentSeries, N::Int)

    # Expand all equations
    equations = []
    for rr in ps.relations
        println("coucou")
    end
end