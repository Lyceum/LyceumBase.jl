# TODO move
macro test_inferred(ex)
    ex = quote
        $Test.@test (($Test.@inferred $ex); true)
    end
    esc(ex)
end

macro test_noalloc(ex)
    ex = quote
        local tmp = $BenchmarkTools.@benchmark $ex samples = 1 evals = 1
        $Test.@test iszero(tmp.allocs)
    end
    esc(ex)
end


const TEST_ALONGS = [
    (True(), ),
    (False(), ),

    # 0 slices
    (False(), False(), False()),
    # 1 slices
    (False(), False(), True()),
    (False(), True(), False()),
    (True(), False(), False()),
    # 2 slices
    (True(), True(), False()),
    (True(), False(), True()),
    (False(), True(), True()),
    # all slices
    (True(), True(), True()),
]

makeA(al::Tuple) = rand(ntuple(i -> i + 1, length(al))...)
makeS(al::Tuple) = Slices(makeA(al), al)
function makeAS(al::Tuple)
    A = makeA(al)
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
        @test_inferred Slices(makeA(al), al)
        @test_inferred Slices(makeA(al), al...)
        @test typeof(Slices(makeA(al), al)) === typeof(Slices(makeA(al), todims(al)))
        @test typeof(Slices(makeA(al), al...)) === typeof(Slices(makeA(al), todims(al)))
        @test typeof(Slices(makeA(al), al)) === typeof(Slices(makeA(al), todims(al)...))
        @test typeof(Slices(makeA(al), al...)) === typeof(Slices(makeA(al), todims(al)...))
    end

    @testset "misc array interface" begin
        A, S = makeAS(al)

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

    @testset "getindex/setindex!" begin
        let
            S = makeS(al)
            x = makeelement(S)
            @test setindex!(S, x, firstindex(S)) === S
        end

        let
            A, S = makeAS(al)
            nested = Array{Array{V, M}, N}(undef, outsize(axes(A), al))
            for i in eachindex(nested)
                nested[i] = makeelement(S)
            end
            for i in eachindex(S, nested)
                S[i] = nested[i]
            end
            @test all(eachindex(S, nested)) do i
                S[i] == nested[i]
            end
        end
    end

    @testset "copy" begin
        S1 = makeS(al)
        S2 = copy(S1)
        @test parent(S1) !== parent(S2)
        @test parent(S1) == parent(S2)
        @test S1.alongs == S2.alongs
        @test S1 == S2
    end

    @testset "copyto!" begin
        S1 = makeS(al)
        S2 = makeS(al)
        @assert S1 !== S2
        @test copyto!(S1, S2) === S1
        @test S1 == S2
    end

    @testset "similar" begin
        A, S = makeAS(al)
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
        S = makeS(al)
        Sv = uview(S)
        @test parent(Sv) isa UnsafeArray{eltype(parent(S)), ndims(parent(S))}
    end

    @testset "parentindices" begin
        S = makeS(al)
        @test all(zip(LinearIndices(S), CartesianIndices(S))) do (i, I)
            parentindices(S, i) == parentindices(S, I) == parentindices(S, Tuple(I)) == parentindices(S, Tuple(I)...)
        end
        @test all(eachindex(S)) do I
            view(parent(S), parentindices(S, I)...) == S[I]
        end
    end

    @testset "functions" begin
        A, S = makeAS(al)
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
    S = Slices(A, True(), False())
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