const TEST_ALONGS = [
    #(static(true), ),
    #(static(false), ),

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

randA(al::Tuple) = rand(ntuple(i -> i + 1, length(al))...)
randS(al::Tuple) = Slices(randA(al), al)
function randAS(al::Tuple)
    A = randA(al)
    S = Slices(A, al)
    return A, S
end

function todims(al::Tuple)
    dims = Int[]
    for (i, a) in enumerate(al)
        SpecialArrays.untyped(a) && push!(dims, i)
    end
    Tuple(dims)
end

function inaxes(parentaxes::Tuple, al::Tuple)
    s = []
    for i = eachindex(parentaxes)
        SpecialArrays.untyped(al[i]) && push!(s, parentaxes[i])
    end
    Tuple(s)
end
outaxes(parentaxes::Tuple, al::Tuple) = inaxes(parentaxes, map(SpecialArrays.not, al))

insize(parentaxes::Tuple, al::Tuple) = map(length, inaxes(parentaxes, al))
outsize(parentaxes::Tuple, al::Tuple) = map(length, outaxes(parentaxes, al))

# Need to construct element with zeros for 0-dimensional elements
makeelement(S::Slices) = rand!(zeros(eltype(parent(S)), size(first(S))...))

@testset "Slices $(todims(al))" for al in TEST_ALONGS
    dims = todims(al)
    L = length(al)
    M = length(dims)
    N = L - M
    V = Float64 # TODO other types?

    @testset "constructors" begin
        @test_inferred Slices(randA(al), al)
        @test_inferred Slices(randA(al), al...)
        @test typeof(Slices(randA(al), al)) === typeof(Slices(randA(al), todims(al)))
        @test typeof(Slices(randA(al), al...)) === typeof(Slices(randA(al), todims(al)))
        @test typeof(Slices(randA(al), al)) === typeof(Slices(randA(al), todims(al)...))
        @test typeof(Slices(randA(al), al...)) === typeof(Slices(randA(al), todims(al)...))
    end

    @testset "misc array interface" begin
        A, S = randAS(al)

        # outer
        @test ndims(S) == N
        @test axes(S) == outaxes(axes(A), al)
        @test size(S) == outsize(axes(A), al)
        @test IndexStyle(S) == IndexCartesian()
        @test Base.dataids(S) == Base.dataids(A)
        @test parent(S) === A

        # inner
        @test ndims(first(S)) == M
        @test axes(first(S)) == inaxes(axes(A), al)
        @test size(first(S)) == insize(axes(A), al)
        @test eltype(S) <: AbstractArray{eltype(A), M}
        @test eltype(S) == typeof(first(S))
    end

    #@testset "getindex/setindex!" begin
    #    let
    #        S = randS(al)
    #        x = Array(first(S))
    #        rand!(x)
    #        @test first(S) != x && @inferred(setindex!(S, x, firstindex(S))) === S && first(S) == x
    #    end

    #    let
    #        #A, S = randAS(al)
    #        for i in eachindex(S, nested)
    #            S[i] = nested[i]
    #        end
    #        @test S == nested
    #        #@test all(eachindex(S, nested)) do i
    #        #    S[i] == nested[i]
    #        #end
    #    end
    #end

    #@testset "copyto!" begin
    #    let
    #        _, flat1 = randNA(U, static(M), static(N))
    #        _, flat2 = randNA(U, static(M), static(N))
    #        dest = NestedView{M}(flat1)
    #        src = NestedView{M}(flat2)
    #        @test copyto!(dest, src) === dest
    #        @test dest == src
    #    end
    #    let
    #        dest, _ = randNA(U, static(M), static(N))
    #        _, flat = randNA(U, static(M), static(N))
    #        src = NestedView{M}(flat)
    #        @test copyto!(dest, src) === dest
    #        @test dest == src
    #    end
    #    let
    #        src, _ = randNA(U, static(M), static(N))
    #        _, flat = randNA(U, static(M), static(N))
    #        dest = NestedView{M}(flat)
    #        @test copyto!(dest, src) === dest
    #        @test dest == src
    #    end
    #end

    @testset "copy" begin
        S1 = randS(al)
        S2 = copy(S1)
        @test parent(S1) !== parent(S2)
        @test parent(S1) == parent(S2)
        @test S1.alongs == S2.alongs
        @test S1 == S2
    end

    @testset "copyto!" begin
        S1 = randS(al)
        S2 = randS(al)
        @assert S1 !== S2
        @test copyto!(S1, S2) === S1
        @test S1 == S2
    end

    @testset "similar" begin
        A, S = randAS(al)
        let B = similar(S)
            @test size(B) == size(S) && eltype(B) === eltype(S)
        end
        let B = similar(S, Int)
            @test size(B) == size(S) && eltype(B) === Int
        end
        let B = similar(S, Int, reverse(size(S)))
            @test size(B) == reverse(size(S)) && eltype(B) === Int
        end
    end

    @testset "UnsafeArrays" begin
        S = randS(al)
        Sv = uview(S)
        @test parent(Sv) isa UnsafeArray{eltype(parent(S)), ndims(parent(S))}
    end

    @testset "parentindices" begin
        S = randS(al)
        @test all(zip(LinearIndices(S), CartesianIndices(S))) do (i, I)
            parentindices(S, i) == parentindices(S, I) == parentindices(S, Tuple(I)) == parentindices(S, Tuple(I)...)
        end
        @test all(eachindex(S)) do I
            view(parent(S), parentindices(S, I)...) == S[I]
        end
    end

    @testset "functions" begin
        A, S = randAS(al)
        @test flatten(S) === A
        @test flatview(S) === A
        @test innersize(S) == size(first(S))
        @test inneraxes(S) == axes(first(S))
        @test_noalloc innersize($S)
        @test_noalloc inneraxes($S)
    end
end

@testset "AxisArrays" begin
    A = AxisArrays.AxisArray(rand(2, 3), [:a, :b], [:x, :y, :z])
    S = Slices(A, static(true), static(false))
    @test S[:x] == A[:, :x]
end

@testset "CartesianIndex" begin
    A = rand(2, 3, 4)
    let
        S = Slices(A, 1)
        @assert ndims(S) == 2
        @test S[CartesianIndex(1)] == S[1]
        @test S[CartesianIndex(1,1)] == S[1]
        @test S[CartesianIndex(2,1)] == S[2]
    end
    let
        S = Slices(A, 1, 2)
        @assert ndims(S) == 1
        @test S[CartesianIndex(1)] == S[1]
        @test S[CartesianIndex(2)] == S[2]
    end
end