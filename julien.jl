include("src/LyceumBase.jl")
#using LyceumBase
#using LyceumBase.SpecialArrays
#using LyceumBase.LyceumCore

using Random
using UnsafeArrays
using StaticNumbers

using AxisArrays: AxisArray

module Mod

using ..LyceumBase.LyceumCore
using Random
using UnsafeArrays
using StaticNumbers
using ..LyceumBase.SpecialArrays

using ..InteractiveUtils
using BenchmarkTools

function _testsplit(xy, bys)
    @code_warntype static_split(xy, bys)
end

function _testmerge(x, y, bys)
    @code_warntype static_merge(x, y, bys)
end

function make(inner, outer)
    S = Array{Array{Float64,length(inner)},length(outer)}(undef, outer...)
    for i in eachindex(S)
        S[i] = rand!(zeros(inner...))
    end
    A = Align(S)
    #F = copy(A)
    @info "" size(first(S)) size(S) size(A) inner outer
    @assert size(S) == outer
    @assert size(first(S)) == inner
    @assert ndims(A) == length(inner) + length(outer)
    A,S
end

function test4()
    L = 4
    M = 3
    N = L - M
    k = M
    inner = ntuple(i -> i, Val(M))
    outer = ntuple(i->(i+M), Val(N))
    @info inner outer
    dims = (inner..., outer...)

    A,S = make(inner, outer)
    return A,S
    F = copy(A)
    #return A,S,F

    I = ntuple(_->1, L)
    @info I
    if A[I...] != F[I...]
        display(A[I...])
        println("----------------")
        display(F[I...])
        display(SpecialArrays._split_indices(A, I))
        @error I
        return A,S,F
    end
    I = ntuple(i -> isodd(i) ? Colon() : dims[i], Val(L))
    @info I

        #return A,S,F
    F2=copy(F)
    @btime $F[$I...] = $A[$I...]
    @btime $F[$I...] = $F2[$I...]
    @btime $F[$I...] = mygetindex($A, $I...)
    if A[I...] != F[I...]
        display(A[I...])
        println("----------------")
        display(F[I...])
        display(SpecialArrays._split_indices(A, I))
        @error I
        return A,S,F
    end
    I = (ntuple(_ -> Colon(), Val(k))..., ntuple(i -> dims[i+k], L - k)...)
    @info I
    if A[I...] != F[I...]
        display(A[I...])
        println("----------------")
        display(F[I...])
        display(SpecialArrays._split_indices(A, I))
        @error I
        return A,S,F
    end
    I = (ntuple(i -> dims[i], k)..., ntuple(_ -> Colon(), L - k)...)
    @info I
    if A[I...] != F[I...]
        display(A[I...])
        println("----------------")
        display(F[I...])
        display(SpecialArrays._split_indices(A, I))
        @error I
        return A,S,F
    end
    I = ntuple(_->Colon(), L)
    @info I
    if A[I...] != F[I...]
        display(A[I...])
        println("----------------")
        display(F[I...])
        display(SpecialArrays._split_indices(A, I))
        @error I
        return A,S,F
    end
    #I = (ntuple(_ -> Colon(), Val(k))..., ntuple(i -> size(S, i+k), N - k)...)
    #I = (ntuple(i -> size(S, i), k)..., ntuple(_ -> Colon(), N - k)...)


    A,S,F
end

function test3()
    #x = (1, 3, 5, 7, 9, 11, 13, 15)
    #y = (2, 4, 6, 8, 10, 12, 14, 16
    #L = 30
    #x = Tuple(i for i=1:L if isodd(i))
    #y = Tuple(i for i=1:L if iseven(i))
    #xy = ntuple(identity, length(x) + length(y))
    #bys = ntuple(i -> isodd(i) ? static(true) : static(false), length(x) + length(y))

    #@info static_filter(bys, xy)
    #@btime static_filter($bys, $xy)
    #@assert static_merge(x, y, bys) == xy
    #@btime static_merge($x, $y, $bys)
    #@info "YO" static_merge(x, y, tail(bys))
    #@info "YO" static_merge(x, tail(y), bys)
    #@info "YO" static_merge(tail(x), y, bys)
    #@info "YO" static_merge(x, y, bys)
    #@btime static_merge($x, $y, $bys)
    #@assert static_split(xy, bys) == reverse(static_split(xy, map(static_not, bys)))
    #@btime static_split($xy, $bys)
    #@btime LyceumCore.static_split2($xy, $bys)
    #static_split(xy, Base.tail(bys))

    #slices = [[1, 2], [3, 4]]
    #aligned = Align(slices, static(false), static(true))
    #@info aligned[1, :] == slices[1]
    #aligned[1, 1] = 0
    #@assert slices == [[0, 2], [3, 4]]
end

function test2()
    L = 3
    M = 1
    N = L - M
    dims = ntuple(identity, Val(L))
    al = ntuple(identity, static(M))
    S = Slices(rand!(zeros(dims...)), al)

    #I = ntuple(identity, Val(N))
    #I = ntuple(_ -> Colon(), Val(N))
    k = 2
    #I = ntuple(i -> isodd(i) ? Colon() : dims[i], Val(N))
    #I = (1, :, 1, :, 1)
    #I = (ntuple(_ -> Colon(), Val(k))..., ntuple(i -> dims[i+k], N - k)...)
    I = (ntuple(_ -> Colon(), Val(k))..., ntuple(i -> size(S, i+k), N - k)...)
    #I = (ntuple(i -> size(S, i), k)..., ntuple(_ -> Colon(), N - k)...)
    #I = (:, 1, :, :, :)
    #I = (:, 1)

    #A = rand(dims[1:N]...)
    #@info I
    #@assert length(I) == ndims(A)
    #V = myview(A, I)
    #@info size(V) IndexStyle(V)
    #return
    #X = [Array(el) for el in S]

    dims = (2,3,4)
    al = (1, )
    S = Slices(rand(dims...), al)
    x = rand(innersize(S)...)
    @info "no slice"
    S[1,1] = x
    @assert S[1,1] == x
    A = [rand!(Array(el)) for el in S[:,1]]
    @info "one slice"
    #S[:,1] = A
    return S
    @assert S[:,1] == A
    A = [rand!(Array(el)) for el in S]
    @info "two slice"
    S[:,:] = A
    @assert S == A
    return S

    IDims = Base.index_dimsum(I...)
    J = parentindices(S, I...)
    x = foo(S,I,J,IDims)
    @info "I" I
    Ilength = length(Base.index_dimsum(I...))
    if Ilength == 0
        @info "case1" x isa AbsArr{<:Any,M}
    elseif Ilength == N
        @info "case2" size(x) size(first(x)) x isa Slices{<:Any,N,M} size(x) == size(S) innersize(x) == innersize(S)
    else
        @info "case3" size(x) size(first(x)) x isa Slices{<:Any,Ilength,M}
    end

    return S
end


function test()
    return
    DIMS = [
        (2, 200),
        (2, 3, 200)
    ]
    AL = [
        (1, ),
        (1, 2),
        (1, 3),
        #(2, 3),
    ]
    for (al, dims) in zip(AL, DIMS)
        A = rand(dims...)
        S = Slices(A, al)
        X = [Array(el) for el in S]
        P = parent(S)
        @info "X S"
        #@btime copyto!($X, $S) evals=1 samples=1
        #@btime copyto2!($X, $S) evals=1 samples=1
        @btime copyto!($X, $S)
        @btime copyto2!($X, $S)
        #@info "S X"
        #@btime copyto!($S, $X) evals=1 samples=1
        #@btime copyto!($X, $S) evals=1 samples=1
        #@btime myycop!($X, $P) evals=1 samples=1
    end

    return

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

end

#S = [rand(2,3) for _=1:4, _=1:5]
L = 5
M = 2
N = L - M
inner = ntuple(i -> 2(1 + i), Val(M))
outer = ntuple(i->2(i+M), Val(N))
A1,S = Mod.make(inner, outer)
A2 = Mod.Align(S, innerfirst=static(true))

#@info "" size(A1) Mod.innersize(A1)
#@info "" size(A2) Mod.innersize(A2)

if false

#for (I,J) in zip(eachindex(A1), eachindex(A2))
#    @assert A1[I] == A2[J]
#end

@info "all int"
I1 = ntuple(i-> firstindex(A1, i), ndims(A1))
I2 = ntuple(i-> firstindex(A2, i), ndims(A2))
#display(getindex(A1, I1...))
#display(Mod.mygetindex(A1, I1...))
@btime $A1[$I1...]

@info "front colon"
I1 = (ntuple(_ -> Colon(), N)..., ntuple(i-> firstindex(A1, N + i), M)...)
I2 = (ntuple(_ -> Colon(), N)..., ntuple(i-> firstindex(A2, N + i), M)...)
#display(getindex(A1, I1...))
#display(Mod.mygetindex(A1, I1...))
@btime $A1[$I1...]

@info "back colon"
I1 = (ntuple(i-> firstindex(A1, i), M)..., ntuple(_ -> Colon(), N)...)
I2 = (ntuple(i-> firstindex(A2, i), M)..., ntuple(_ -> Colon(), N)...)
#display(getindex(A1, I1...))
#display(Mod.mygetindex(A1, I1...))
@btime $A1[$I1...]

end

function testone(A, x, I...)
    @info I
    A[I...] = x
    @assert A[I...] == x

    A = Mod.Align(A.slices, innerfirst=static(true))
end

if true
    S = [rand(2,3) for _=1:4, _=1:5]
    #@info "" size(S) size(first(S))
    #F = reshape(mapreduce(vcat, hcat, S), (size(first(S))..., size(S)...));
    A = Mod.Align(S, innerfirst=static(false))
    testone(A, 100, 4, 5, 2, 3)
    testone(A, rand(2), 4, 5, :, 3)
    testone(A, rand(3), 4, 5, 2, :)
    testone(A, rand(2,3), 4, 5, :, :)
    testone(A, rand(4,5), :, :, 2, 3)
    testone(A, rand(4,5,3), :, :, 2, :)
    testone(A, rand(size(A)...), :, :, :, :)

    A = Mod.Align(S, innerfirst=static(true))
    testone(A, 100, 2, 3, 4, 5)
    testone(A, rand(4), 2, 3, :, 5)
    testone(A, rand(5), 2, 3, 4, :)
    testone(A, rand(4,5), 2, 3, :, :)
    testone(A, rand(2,3), :, :, 4, 5)
    testone(A, rand(2,3,5), :, :, 4, :)
    testone(A, rand(size(A)...), :, :, :, :)
    #@btime $F[:,:,4,5,6] = Mod.mygetindex($A, :, :, 4, 5, 6)
    #@btime $F[:,:,4,5,6] = Mod.mygetindex2($A, :, :, 4, 5, 6)
end

nothing