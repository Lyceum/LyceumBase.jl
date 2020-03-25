struct CartesianIndexer{T,N,A<:AbsArr{T,N}} <: AbstractArray{T,N}
    parent::A
end

@forward CartesianIndexer.parent Base.size, Base.axes, Base.length

@propagate_inbounds function Base.getindex(A::CartesianIndexer{<:Any,N}, I::Vararg{Int,N}) where {N}
    getindex(A.parent, I...)
end

@propagate_inbounds function Base.setindex!(A::CartesianIndexer{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    setindex!(A.parent, v, I...)
end

Base.IndexStyle(::Type{<:CartesianIndexer}) = IndexCartesian()

function Base.similar(A::CartesianIndexer, T::Type, dims::Dims)
    similar(A.parent, T, dims)
end

