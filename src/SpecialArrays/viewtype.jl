# For AbstractArray A and indices I, compute V = typeof(view(A, I...)) by:
# 1) Try to infer V from typeof(A) and typeof(I) by calling viewtype(typeof(A), typeof(I))
# 2) If !isconcretetype(V), fall back to typeof(view(A, I...))
# Custom subtypes of AbstractArray (e.g. UnsafeArray) could provide customized implementations
# of viewtype(A::AbstractArray, I::Tuple) or viewtype(::Type{<:AbstractArray}, ::Type{<:Tuple})
# if required.

@inline viewtype(A::AbstractArray, I::Tuple) = _viewtype(A, I)
@inline viewtype(A::AbstractArray, I...) = _viewtype(A, I)

@generated function _viewtype(A::AA, I::II) where {AA<:AbstractArray,II<:Tuple}
    T = Core.Compiler.return_type(view, Tuple{AA, II.parameters...})
    if isconcretetype(T)
        return :(Base.@_inline_meta; $T)
    else
        return :(Base.@_inline_meta; __viewtype(A, I))
    end
end

@inline function __viewtype(A::AbstractArray, I::Tuple)
    try
        typeof(@inbounds view(A, I...))
    catch e
        Istring = join(map(string, I), ", ")
        printstyled(stderr,
        """
        Unable to infer typeof(view($(typeof(A)), $(join(map(i -> string(typeof(i)), I), ", "))))
        Only other option is to try typeof(view(A, $Istring)) but that resulted in below error.
        Try passing in valid indices or implement:
            viewtype(::Type{$(typeof(A))}, ::Type{$(typeof(I))})
        """, color=:light_red)
        rethrow(e)
    end
end