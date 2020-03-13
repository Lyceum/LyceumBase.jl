# For AbstractArray A and indices I, compute typeof(view(A, I...)) by:
# 1) First trying to infer T from typeof(A) and typeof(I). This avoids indexing into
#   A, which is faster and allows for 0-length axes
# 2) If that fails, return typeof(view(A, I...))

@propagate_inbounds viewtype(A::AbsArr, I...) = viewtype(A, I)
@propagate_inbounds function viewtype(A::AbsArr, I::Tuple)
    T = viewtype(typeof(A), typeof(I))
    _viewtype(A, I, T, Val(isconcretetype(T)))
end

@inline function viewtype(::Type{A}, ::Type{I}) where {A<:AbsArr,I<:Tuple}
    Core.Compiler.return_type(view, _view_signature(A, I))
end
@pure _view_signature(::Type{A}, ::Type{I}) where {A,I<:Tuple} = Tuple{A, I.parameters...}

_viewtype(A, I, T, ::Val{true}) = T
@propagate_inbounds _viewtype(A, I, T, ::Val{false}) = typeof(view(A, I...))
