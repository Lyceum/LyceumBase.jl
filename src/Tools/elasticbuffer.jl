abstract type AbstractElasticBuffer{names,T,A} <: AbstractVector{NamedTuple{names,T}} end

const DEFAULT_CAPACITY = 1024

fieldarrays(b::AbstractElasticBuffer) = getfield(b, :fieldarrays)

Base.length(b::AbstractElasticBuffer) = getfield(b, :length)
Base.size(b::AbstractElasticBuffer) = (length(b),)
Base.IndexStyle(b::AbstractElasticBuffer) = IndexLinear()
Base.getproperty(b::AbstractElasticBuffer, name::Symbol) =
    view(getproperty(fieldarrays(b), name), .., Base.OneTo(length(b)))
Base.propertynames(b::AbstractElasticBuffer{names}) where {names} = names

Base.@propagate_inbounds function Base.getindex(
    b::AbstractElasticBuffer,
    I::Union{Int,UnitRange{Int}},
)
    @boundscheck checkbounds(b, I)
    mapfield(x -> x[.., I], fieldarrays(b))
end

Base.@propagate_inbounds function Base.setindex!(
    b::AbstractElasticBuffer{names},
    vals::NamedTuple{names},
    I::Int,
) where {names}
    @boundscheck (checkbounds(b, I); checksetindex(b, vals))
    foreachfield_tabular(
        (x, v) -> (@inbounds copyto!(uview(x, .., I), v)),
        (fieldarrays(b), vals),
    )
    b
end


mutable struct ElasticBuffer{names,T,A} <: AbstractElasticBuffer{names,T,A}
    fieldarrays::NamedTuple{names,A}
    length::Int
    capacity::Int
    function ElasticBuffer{names,T,A}(
        fieldarrays::NamedTuple{names,A},
        length::Int,
        capacity::Int,
    ) where {names,T,A}
        check_parameters(T, A)
        b = new{names,T,A}(fieldarrays, length, capacity)
        checkdims(b)
        b
    end
end

@generated function ElasticBuffer(
    fieldarrays::NamedTuple{names,A},
    length::Int,
    capacity::Int,
) where {names,A}
    T = Tuple{map(getT, Tuple(A.parameters))...}
    :(ElasticBuffer{$names,$T,$A}(fieldarrays, length, capacity))
end

function ElasticBuffer(length::Int, shapes::NamedTuple; capacity::Int = DEFAULT_CAPACITY)
    fieldarrays = map(s -> allocatebuffer(s, capacity), shapes)
    ElasticBuffer(fieldarrays, length, capacity)
end
ElasticBuffer(shapes::NamedTuple; capacity::Int = DEFAULT_CAPACITY) =
    ElasticBuffer(0, shapes, capacity = capacity)

ElasticBuffer(length::Int = 0, capacity::Int = DEFAULT_CAPACITY; shapes...) =
    ElasticBuffer(length, values(shapes), capacity = capacity)
capacity(b::ElasticBuffer) = getfield(b, :capacity)
Base.empty!(b::ElasticBuffer) = (setfield!(b, :length, 0); checkdims(b); b)

function Base.sizehint!(b::ElasticBuffer, n::Integer)
    if capacity(b) != n
        foreachfield(x -> elasticresize!(x, n), fieldarrays(b))
        setfield!(b, :capacity, n)
        checkdims(b)
    end
    b
end

grow!(b::ElasticBuffer, n::Integer = 1) =
    (resize!(b, length(b) + n); checkdims(b); return b)

function Base.resize!(b::ElasticBuffer, n::Integer)
    mayberesize!(b, n)
    setfield!(b, :length, n)
    checkdims(b)
    b
end

function mayberesize!(b::ElasticBuffer, n::Int)
    if n >= capacity(b)
        n = nextpow(2, nextpow(2, n))
        sizehint!(b, n)
        checkdims(b)
    end
    b
end



struct ElasticBufferView{names,T,A} <: AbstractElasticBuffer{names,T,A}
    fieldarrays::NamedTuple{names,A}
    length::Int
    function ElasticBufferView{names,T,A}(
        fieldarrays::NamedTuple{names,A},
        length::Int,
    ) where {names,T,A}
        check_parameters(T, A)
        b = new{names,T,A}(fieldarrays, length)
        checkdims(b)
        b
    end
end

@generated function ElasticBufferView(
    fieldarrays::NamedTuple{names,A},
    length::Int,
) where {names,A}
    T = Tuple{map(getT, Tuple(A.parameters))...}
    :(ElasticBufferView{$names,$T,$A}(fieldarrays, length))
end

@generated function UnsafeArrays.unsafe_uview(b::ElasticBuffer{names,T,A}) where {names,T,A}
    ex = Expr(:tuple)
    for i = 1:length(names)
        push!(ex.args, :(uview(getfield(fa, $i), .., 1:l)))
    end
    quote
        l = length(b)
        fa = fieldarrays(b)
        ElasticBufferView(NamedTuple{$names}($ex), l)
    end
end

Base.@propagate_inbounds function Base.unsafe_view(
    b::AbstractElasticBuffer{names,T,A},
    I::UnitRange{Int},
) where {names,T,A}
    @boundscheck checkbounds(b, I)
    v = mapfield(x -> view(x, .., I), fieldarrays(b)) #TODO
    ElasticBufferView(v, length(I))
end

Base.@propagate_inbounds function Base.unsafe_view(
    b::AbstractElasticBuffer{names,T,A},
    I::Int,
) where {names,T,A}
    @boundscheck checkbounds(b, I)
    v = mapfield(x -> view(x, .., I:I), fieldarrays(b))
    ElasticBufferView(v, 1)
end



function checksetindex(b::AbstractElasticBuffer{names}, vals::NamedTuple{names})
    fa = fieldarrays(b)
    foreachfield_tabular((fa, vals)) do x, v
        Base.front(size(
            x,
        )) == size(v) || error("size of val doesn't match fieldarray value")
    end
    nothing
end

function checkdims(b::ElasticBufferView)
    if length(b) > 0
        foreachfield(fieldarrays(b)) do x
            if ndims(x) > 1
                size(
                    x,
                    ndims(x),
                ) == length(
                    b,
                ) || error("fieldarrays must have size(x, ndims(x)) == length(ElasticBufferView)")
            end
        end
    end
    length(b) >= 0 || throw(DomainError(length(b), "Invalid length."))
    nothing
end


function checkdims(b::ElasticBuffer)
    foreachfield(fieldarrays(b)) do x
        size(
            x,
            ndims(x),
        ) == capacity(
            b,
        ) || error("fieldarrays must have size(x, ndims(x)) == capacity(ElasticBuffer)")
    end
    length(b) <= capacity(b) || error("length greater than capacity")
    length(b) >= 0 || throw(DomainError(length(b), "Invalid length."))
    capacity(b) >= 0 || throw(DomainError(capacity(b), "Invalid capacity."))
    nothing
end

@generated function check_parameters(::Type{T}, ::Type{A}) where {T,A}
    length(T.parameters) == length(A.parameters) || return :(error("Types are different lengths"))
    for (t, a) in zip(T.parameters, A.parameters)
        expected = getT(a)
        errmsg = "Type mismatch. Expected $expected, got $t."
        if a isa DataType
            expected == t || return :(error($errmsg))
        else
            t <: AbstractArray && a <: AbstractArray && expected == t ||
            return :(error($errmsg))
        end
    end
end

resizedims(x::ElasticArray, newdim::Int) = (Base.front(size(x))..., newdim)
function elasticresize!(x::ElasticArray, newdim::Int)
    resize!(x, resizedims(x, newdim)...)
end

allocatebuffer(shape::AbstractShape, n::Int) = ElasticArray(undef, shape, n)
allocatebuffer(T::DataType, n::Int) = ElasticArray{T}(undef, n)

getT(::Type{<:ElasticArray{T,1,0}}) where {T} = T
getT(::Type{<:ElasticArray{T,N,M}}) where {T,N,M} = Array{T,M}
getT(::Type{<:AbstractArray{T,1}}) where {T,N} = T
getT(::Type{<:AbstractArray{T,N}}) where {T,N} = Array{T,N - 1}
getT(::Any) = nothing
