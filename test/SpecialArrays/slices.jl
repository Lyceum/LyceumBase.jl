const TEST_ALONGS = [
    (static(true), ),
    (static(false), ),

    ## 0 slices
    #(static(false), static(false), static(false)),
    ## 1 slices
    #(static(false), static(false), static(true)),
    #(static(false), static(true), static(false)),
    #(static(true), static(false), static(false)),
    ## 2 slices
    #(static(true), static(true), static(false)),
    #(static(true), static(false), static(true)),
    (static(false), static(true), static(true)),
    # all slices
    #(static(true), static(true), static(true)),
]


randA(T::Type, al::NTuple{N,StaticBool}) where {N} = randA(T, static(N))
randA(al::TupleN{StaticBool}) = randA(DEFAULT_ELTYPE, al)

randS(T::Type, al::TupleN{StaticBool}) = Slices(randA(T, al), al)
randS(al::TupleN{StaticBool}) = randS(DEFAULT_ELTYPE)

randAS(T::Type, al::TupleN{StaticBool}) = (A = randA(T, al); return A, Slices(A, al))
randAS(al::TupleN{StaticBool}) = randAS(DEFAULT_ELTYPE, al)

function randN(T::Type, al::NTuple{N,StaticBool}) where {N}
    dims = testdims(N)
    innersz = Tuple(dims[i] for i in 1:length(al) if unstatic(al[i]))
    outersz = Tuple(dims[i] for i in 1:length(al) if !unstatic(al[i]))
    return randN(T, innersz, outersz)
end
randN(al::TupleN{StaticBool}) = randN(DEFAULT_ELTYPE, al)



@testset "Slices $V $(slicedims(al))" for V in (Float64, ), al in TEST_ALONGS
    sdims = slicedims(al)
    dims = map(unstatic, sdims)

    inaxes, outaxes = let A = randA(V, al)
        inaxes = Tuple(axes(A, i) for i in 1:length(al) if unstatic(al[i]))
        outaxes = Tuple(axes(A, i) for i in 1:length(al) if !unstatic(al[i]))
        inaxes, outaxes
    end
    insize = map(length, inaxes)
    outsize = map(length, outaxes)

    M = length(insize)
    N = length(outsize)
    ST = Slices{<:AbsArr{V,M},N,M,Array{V,M+N},typeof(al)}


    @testset "constructors" begin
        @test typeof(@inferred(Slices(randA(al), al))) <: ST
        @test typeof(@inferred(Slices(randA(al), al...))) <: ST
        @test typeof(Slices(randA(al), dims)) <: ST
        @test typeof(Slices(randA(al), dims...)) <: ST
        @test typeof(@inferred(Slices(randA(al), sdims))) <: ST
        @test typeof(@inferred(Slices(randA(al), sdims...))) <: ST
    end

    @testset "misc array interface" begin
        A, S = randAS(al)

        # outer
        @test eltype(S) <: AbsArr{V,M}
        @test ndims(S) == N
        @test @inferred(axes(S)) == outaxes
        @test @inferred(size(S)) == outsize
        @test length(S) == prod(outsize)
        @test @inferred(IndexStyle(S)) == IndexCartesian()
        @test @inferred(Base.dataids(S)) == Base.dataids(A)
        @test @inferred(parent(S)) === A

        # inner
        @test eltype(first(S)) == V
        @test ndims(first(S)) == M
        @test axes(first(S)) == inaxes
        @test size(first(S)) == insize
        @test length(first(S)) == prod(insize)
        @test typeof(first(S)) == eltype(S)
    end

    @testset "getindex/setindex!" begin
        let S = randS(V, al), x = rand!(zeros(V, insize...))
            @test @inferred(setindex!(S, x, firstindex(S))) === S && first(S) == x
        end

        let S = randS(V, al), nested = randN(V, al)
            for I in eachindex(S, nested)
                S[I] = nested[I]
            end
            @test all(zip(LinearIndices(S), CartesianIndices(S))) do (I, J)
                S[I] == S[J] == nested[I]
            end
        end

        # Colon
        let S = randS(V, al), B = @inferred(S[:])
            @test typeof(B) <: Array{<:AbsArr{V,M},N}
            @test size(S) == size(B)
            @test innersize(S) == innersize(B)
        end
        let S = randS(V, al), B = [rand!(similar(el)) for el in S]
            S[:] = B
            @test S == B
        end
    end

    @testset "similar" begin
        S = randS(V, al)
        U = V === Float64 ? Int : Float64

        let B = @inferred(similar(S))
            @test typeof(B) <: Slices{<:AbsArr{V,M},N,M,Array{V,M+N}}
            @test size(B) == size(S)
            @test size(first(B)) == size(first(S))
            @test B.alongs == Tuple(sort([el for el in S.alongs], rev=true))
        end
        let B = @inferred(similar(S, Array{U})), C = @inferred(similar(S, U))
            @test typeof(B) == typeof(C)
            @test typeof(B) <: Slices{<:AbsArr{U,M},N,M,Array{U,M+N}}
            @test size(B) == size(C) == size(S)
            @test size(first(B)) == size(first(C)) == size(first(S))
            @test B.alongs == Tuple(sort([el for el in S.alongs], rev=true))
        end
        if N > 1
            newdims = reverse(size(S))
            let B = @inferred(similar(S, newdims))
                @test typeof(B) <: Slices{<:AbsArr{U,M},N,M,Array{U,M+N}}
                @test size(B) == newdims
                @test size(first(B)) == size(first(S))
                @test B.alongs == Tuple(sort([el for el in S.alongs], rev=true))
            end
            let B = @inferred(similar(S, Array{U}, newdims)), C = @inferred(similar(S, U, newdims))
                @test typeof(B) == typeof(C)
                @test typeof(B) <: Slices{<:AbsArr{U,M},N,M,Array{U,M+N}}
                @test size(B) == size(C) == newdims
                @test size(first(B)) == size(first(C)) == size(first(S))
                @test B.alongs == C.alongs == Tuple(sort([el for el in S.alongs], rev=true))
            end
        end
    end

    @testset "copyto!" begin
        # size(dest) == size(src)
        let dest = randS(V, al), src = randS(V, al)
            @test copyto!(dest, src) === dest
            @test dest == src
        end
        let dest = randS(V, al), src = randN(V, al)
            @test copyto!(dest, src) === dest
            @test dest == src
        end
        let dest = randN(V, al), src = randS(V, al)
            @test copyto!(dest, src) === dest
            @test dest == src
        end

        # size(dest) != size(src)
        if N != 1
            let dest = randS(V, al), src = vec(randN(V, al))
                @test copyto!(dest, src) === dest
                N == 0 ? @test(dest[] == first(src)) : @test(dest == src)
            end
            let dest = vec(randN(V, al)), src = randS(V, al)
                @test copyto!(dest, src) === dest
                N == 0 ? @test(dest[] == first(src)) : @test(dest == src)
            end
        end
    end

    @testset "copy" begin
        S1 = randS(V, al)
        S2 = copy(S1)
        @test parent(S1) !== parent(S2)
        @test parent(S1) == parent(S2)
        @test S1.alongs == S2.alongs
        @test S1 == S2
    end

    @testset "functions" begin
        S = randS(V, al)

        let B = @inferred(flatten(S))
            @test B !== parent(S) && B == parent(S)
        end
        @test @inferred(flatview(S)) === parent(S)

        @test @inferred(innersize(S)) == size(first(S))
        @test @inferred(inneraxes(S)) == axes(first(S))
        @test_noalloc innersize($S)
        @test_noalloc inneraxes($S)

        @test Slices(parent(S), @inferred(slicedims(S))) === S
    end

    @testset "UnsafeArrays" begin
        S = randS(V, al)
        Sv = uview(S)
        @test parent(Sv) isa UnsafeArray{eltype(parent(S)), ndims(parent(S))}
        @test S == Sv
    end

    @testset "parentindices" begin
        S = randS(V, al)
        @test all(zip(LinearIndices(S), CartesianIndices(S))) do (i, I)
            parentindices(S, i) == parentindices(S, I) == parentindices(S, Tuple(I)...)
        end
    end
end

@testset "non-standard indexing (AxisArrays)" begin
    A = AxisArrays.AxisArray(rand(2, 3), [:a, :b], [:x, :y, :z])
    S = Slices(A, static(true), static(false))
    @test S[:x] == A[:, :x]
end
