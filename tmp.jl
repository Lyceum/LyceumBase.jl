


#f(A) = Slices(A, (static(1), static(2), static(5)))
#g(A) = Slices(A, (1, 2, 5))
#using Base: @pure
#
#@inline function foo(A::AbsArr{T,N}, al::Dims{M}) where {T,N,M}
#    #ntuple(dim -> foo(dim, al), Val(N))
#    ntuple(dim -> foo(dim, al), Val(N))
#end
#
#@pure foo(d::Int, al::Dims{M}) where {M} = static(d in al)
#
#using Base.Broadcast: Broadcasted, instantiate, throwdm, preprocess
#function broad(S,A)
#    bc = Base.broadcasted(identity, A)
#    mymat!(S, bc)
#end
#
#function mymat!(dest, bc::Broadcasted{Style}) where {Style}
#    inst = instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest)))
#    x = convert(Broadcasted{Nothing}, inst)
#    mycopyto!(dest, x)
#end
#
#
#
## DEFAULT
#@inline function mycopyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
#    @info "SWAG"
#    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
#    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
#    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
#        A = bc.args[1]
#        if axes(dest) == axes(A)
#            @info "BAM"
#            return copyto!(dest, A)
#        end
#    end
#    bc′ = preprocess(dest, bc)
#    @simd for I in eachindex(bc′)
#        @inbounds dest[I] = bc′[I]
#    end
#    return dest
#end
#
#function sim()
#    A = rand(2,3,4)
#    al = (1, 3)
#    S = Slices(A, al)
#
#    B = similar(S)
#    B = similar(S, Array{Int})
#    B = similar(S, Int)
#    B = similar(S, Array{Int}, (5, 6))
#    B = similar(S, Int, (3, ))
#end

#end




using Base: index_shape, index_dimsum, unsafe_length, throw_checksize_error
using Base: _getindex, _unsafe_getindex, _unsafe_getindex!
using InteractiveUtils
using Base.Cartesian
using MacroTools
# Always index with the exactly indices provided.
#@generated function my_unsafe_getindex!(dest::AbstractArray, src::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
function damgetindex(A, I...)
    @info I
    getindex(A, I...)
end
function my_unsafe_getindex!(dest::AbstractArray, src::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
    D = eachindex(dest)
    Dy = iterate(D)
    @info "YOasfasfgasfd" I
    #x=prettify(@macroexpand @nloops 2 j d->I[d] begin
    #    # This condition is never hit, but at the moment
    #    # the optimizer is not clever enough to split the union without it
    #    Dy === nothing && return dest
    #    (idx, state) = Dy
    #    dest[idx] = @ncall 2 damgetindex src j
    #    Dy = iterate(D, state)
    #end)
    #display(x)
    @nloops 2 j d->I[d] begin
        # This condition is never hit, but at the moment
        # the optimizer is not clever enough to split the union without it
        Dy === nothing && return dest
        (idx, state) = Dy
        dest[idx] = @ncall 2 damgetindex src j
        Dy = iterate(D, state)
    end
    return dest
end
function my_unsafe_getindex(::IndexStyle, A::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
    # This is specifically not inlined to prevent excessive allocations in type unstable code
    shape = Base.index_shape(I...)
    dest = similar(A, shape)
    map(unsafe_length, axes(dest)) == map(unsafe_length, shape) || throw_checksize_error(dest, shape)
    @info size(dest) I
    #@edit _unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    my_unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    return dest
end

A=rand(2,3)
I=(:,1)
J=Base.to_indices(A, I)
K = A[:,]
l=IndexStyle(A)
#@edit Base._getindex(IndexStyle(A), A, J...)
#@edit _unsafe_getindex(l, Base._maybe_reshape(l, A, J...), J...)


#B=my_unsafe_getindex(l, Base._maybe_reshape(l, A, J...), J...)