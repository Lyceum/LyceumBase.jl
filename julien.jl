include("src/LyceumBase.jl")
using Random
using UnsafeArrays
using StaticNumbers

#using .LyceumBase.SpecialArrays
#using .LyceumBase.SpecialArrays: NestedView

module Mod

using ..LyceumBase.LyceumCore
using Random
using UnsafeArrays
using StaticNumbers
using ..LyceumBase.SpecialArrays

function dam(parent::AbsArr{T,N}, alongs::Vararg{StaticInteger,M}) where {T,N,M}
    #alongs = map(dim -> dim in alongs ? True() : False(), ntuple(identity, Val(N)))
    Slices(parent, alongs)
end

function dam(parent::AbsArr{T,N}, alongs::Vararg{StaticBool,M}) where {T,N,M}
    Slices(parent, alongs)
end
using ..InteractiveUtils
using BenchmarkTools

function foo!(A, B)
    @inbounds for I = eachindex(A,B)
        A[I] = B[I]
    end
end

function test()
    dims = (2,3,4)
    #dims = ntuple(_ -> rand(1:2), 20)
    #al = ntuple(i -> isodd(i) ? static(true) : static(false), length(dims))
    al1 = (1, 3)
    al2 = (1, 3)
    A = rand(dims...)
    S = Slices(A, al1)
    X = [Array(el) for el in S]
    return S,X

    S[1]
    S[I...]
    S[CartesianIndex(I...)]
    S[1] = x
    S[I...] = x
    S[CartesianIndex(I...)] = x





    S1 = Slices(rand(dims...), al1)
    S2 = Slices(rand(dims...), al2)
    return S1,S2
    X = [Array(el) for el in S]

    #@btime SpecialArrays._pidxs($S, $I...)
    #@btime parentindices($S, $I...)
    #foo!(S1,S2)
    #@btime foo!($S1, $X)

    if false
    rand!(S1.parent)
    @assert S1 != S2
    copyto!(S1, S2)
    @info "EQUAL S S" S1 == S2

    rand!(S1.parent)
    @assert X != S1
    copyto!(X, S1)
    @info "EQUAL X S" X == S1

    rand!(S1.parent)
    @assert S1 != X
    copyto!(S1, X)
    @info "EQUAL S X" S1 == X
    end

    if false
    @btime copyto!($S1, $S2) evals=1 samples=5
    @btime copyto!($X, $S2) evals=1 samples=5
    @btime copyto!($S1, $X) evals=1 samples=5

    @uviews S1 S2 X begin
    @btime copyto!($S1, $S2) evals=1 samples=5
    @btime copyto!($X, $S2) evals=1 samples=5
    @btime copyto!($S1, $X) evals=1 samples=5
    end
    end

    return S1,S2
end

f(A) = Slices(A, (static(1), static(2), static(5)))
g(A) = Slices(A, (1, 2, 5))
using Base: @pure

@inline function foo(A::AbsArr{T,N}, al::Dims{M}) where {T,N,M}
    #ntuple(dim -> foo(dim, al), Val(N))
    ntuple(dim -> foo(dim, al), Val(N))
end

@pure foo(d::Int, al::Dims{M}) where {M} = static(d in al)

using Base.Broadcast: Broadcasted, instantiate, throwdm, preprocess
function broad(S,A)
    bc = Base.broadcasted(identity, A)
    mymat!(S, bc)
end

function mymat!(dest, bc::Broadcasted{Style}) where {Style}
    inst = instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest)))
    x = convert(Broadcasted{Nothing}, inst)
    mycopyto!(dest, x)
end

# DEFAULT
@inline function mycopyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
    @info "SWAG"
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            @info "BAM"
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end

function sim()
    A = rand(2,3,4)
    al = (1, 3)
    S = Slices(A, al)

    B = similar(S)
    B = similar(S, Array{Int})
    B = similar(S, Int)
    B = similar(S, Array{Int}, (5, 6))
    B = similar(S, Int, (3, ))
end

end