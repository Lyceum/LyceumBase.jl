# For a::A and idxs::I, where A <: AbstractArray{<:Any, N} and I <: Tuple{Vararg{<:Any, N}}
# compute T = typeof(view(a, idxs...))
# 1) First try T = viewtype(A, I)
# 2) If that fails, try to infer the return type from A and I.
# 2) If that does not yield a concrete T and all the axes of A have non-zero length,
#    try T = typeof(view(a, idxs...))

# Main entry point
@inline viewtype(a::AbstractArray, idxs...) = viewtype(a, idxs)
@inline function viewtype(a::AbstractArray{<:Any, N}, idxs::Tuple{Vararg{<:Any, N}}) where {N}
    _viewtype(a, idxs, viewtype(typeof(a), typeof(idxs)))
end

@inline unsafe_viewtype(a::AbstractArray, idxs...) = unsafe_viewtype(a, idxs)
@inline function unsafe_viewtype(A::AbstractArray{<:Any, N}, idxs::Tuple{Vararg{<:Any, N}}) where {N}
    @inbounds typeof(view(A, idxs...))
end

# Fall back
@inline function viewtype(A::Type{<:AbstractArray{<:Any, N}}, I::Type{<:Tuple{Vararg{<:Any, N}}}) where {N}
    _try_infer_viewtype(A, I)
end

_viewtype(a, idxs, T) = T
@propagate_inbounds function _viewtype(a, idxs, ::Nothing)
    @boundscheck begin
        inbounds = checkbounds(Bool, a, idxs...)
        !inbounds && throw(ArgumentError(
            """
            Unable to infer the type of T, where T = viewtype(::Type{$(typeof(A))}, ::Type{$(typeof(idxs))})
            Only other option is to perform a view into `A`, but `idxs` $idxs are invalid.
            Try passing in valid indices or implement:
                viewtype(::Type{$(typeof(A))}, ::Type{$(typeof(idxs))}) -> T
            """
        ))
    end
    @inbounds unsafe_viewtype(a, idxs)
end

@pure _view_signature(::Type{A}, ::Type{I}) where {A,I<:Tuple} = Tuple{A, I.parameters...}
@inline function _try_infer_viewtype(::Type{A}, ::Type{I}) where {A,I<:Tuple}
    T = Core.Compiler.return_type(view, _view_signature(A, I))
    isconcretetype(T) ? T : nothing
end