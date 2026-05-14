using Symbolics

#-------------------------------------------------------------PowerSeries--------------------------------------------------------

"""
    PowerSeries{T,D}

    An abstract type to represent algebraic multivariate, multidimensional power series.
     
    T is the type of the coefficients
    D is the number of dimensions (0 = scalar series, 1 = vector series, ...)

    ### Notes

    Every concrete PowerSeries{T,D} must have the following fields and methods :
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
abstract type PowerSeries{T,D} end  

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
    a Vector{T} if Didx == Dps or a T if Didx == Dps+1
"""
function Base.getindex(ps::PowerSeries{Dps}, I::Vararg{Int64,Didx}) where {Dps,Didx}
    if Didx <= Dps
        ps.coefficients[I...]
    elseif Didx == Dps+1
        ps.coefficients[I[begin:end-1]...][I[end]+1]
    else 
        throw(DimensionMismatch("Index dimension is above maximum index dimension for the PowerSeries"))
    end
end

#-----------------------------------------------------------TaylorSeries-------------------------------------------------------------


"""
    TaylorSeries{T,D} <: PowerSeries{T,D}

    A concrete type representing the Taylor development of a function around a center c

    ### Fields
    - `size::NTuple{D,Int}` -- size of the series (similar to Array)
    - `variables::Vector{Num}` -- The series variables, for instance x in Σaᵢxⁱ
    - `center::Vector` -- The series center, i.e c in Σaᵢ(x-c)ⁱ
    - `coefficients::Array{Vector{T},D}` -- The series coefficients
    - `order::Int` -- The order to which coefficients were already computed (-1 means none)
    - `func::Num` -- Symbolics representation of a function of the variables
    - `last_computed_derivatives::Vector{Num}` -- Symbolics representation of the last 
      derivatives of func that were computed to compute the coefficients
    - `origin_eval_dict::Dict{Num, T}` -- A dict to associate each symbol to its value at 
      the center of the series
    - `differentials::Vector{Differential}` -- A vector to store the differential operators
      with respect to the different variables
    - `factorials::Vector{Int}` -- A vector to store the factorial coefficients to apply
      to each differential when computing coefficients
    - `factorials_orders::Vector{Vector{Int}}` -- A vector to store the corresponding orders of
      each variable in the factorials vector

    ### Examples
    - `TaylorSeries{T}(variables::Vector{Num}, 
                       func::Num,
                       center::Vector, 
                       size::NTuple{D, Int}=(1,))` -- default constructor
"""
mutable struct TaylorSeries{T,D} <: PowerSeries{T,D}
    
    size::NTuple{D,Int}
    variables::Vector{Num}
    center::Vector
    coefficients::Array{Vector{T},D}
    order::Int
    func::Num

    # used for coefficients computation
    last_computed_derivatives::Vector{Num}
    origin_eval_dict::Dict{Num, T}
    differentials::Vector{Differential}
    factorials::Vector{Int}
    factorials_orders::Vector{Vector{Int}}
end

function TaylorSeries{T}(variables::Vector{Num}, 
                        func::Num, 
                        center::Vector, 
                        size::NTuple{D, Int}=(1,)) where {T,D}
    if length(variables)==length(center)
        origin_eval_dict = Dict([v=>(c |> T) for (v,c) in zip(variables, center)])
        differentials = [Differential(v) for v in variables]
        # create coefficients array
        coeffs = Array{Vector{T}}(undef, size)
        for i in eachindex(coeffs)
            coeffs[i] = Vector{T}()
        end
        TaylorSeries(size, variables, center, coeffs, -1, func, Num[], origin_eval_dict, 
                     differentials, Int[], Vector{Int}[])
    else
        throw(ArgumentError("center size does not match number of variables"))
    end
end

"""
    compute_coefficients(ps::TaylorSeries, N::UInt)

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
        last_computed_derivatives = ps.last_computed_derivatives
        factorials = ps.factorials
        factorials_orders = ps.factorials_orders
        if isempty(last_computed_derivatives)
            ps.last_computed_derivatives = [ps.func]
            ps.factorials = [1]
            ps.factorials_orders = [zeros(Int, length(ps.variables))]
        else 
            ps.last_computed_derivatives = []
            ps.factorials = []
            ps.factorials_orders = Vector{Int}[]
            for (d,f,fo) in zip(last_computed_derivatives, factorials, factorials_orders)
                for diff in ps.differentials
                    push!(ps.last_computed_derivatives, expand_derivatives(diff(d)))
                end
                for vidx in eachindex(ps.variables)
                    push!(ps.factorials, f*(fo[vidx]+1))
                    new_order = copy(fo)
                    new_order[vidx] += new_order[vidx] + 1
                    push!(ps.factorials_orders,new_order)
                end
            end
        end

        # compute coefficients of order N
        for (func,f) in zip(ps.last_computed_derivatives, ps.factorials)
            coeff = substitute(func, ps.origin_eval_dict)./f
            for i in eachindex(ps.coefficients)
                val = Symbolics.value(coeff[i])
                push!(ps.coefficients[i], (val |> T))
            end
        end
    end
end




#------------------------------------------------------------RecursiveSeries----------------------------------------------------