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


randA(T::Type, al::NTuple{N,SBool}) where {N} = randA(T, static(N))
randA(al::TupleN{SBool}) = randA(DEFAULT_ELTYPE, al)

randS(T::Type, al::TupleN{SBool}) = Slices(randA(T, al), al)
randS(al::TupleN{SBool}) = randS(DEFAULT_ELTYPE)

randAS(T::Type, al::TupleN{SBool}) = (A = randA(T, al); return A, Slices(A, al))
randAS(al::TupleN{SBool}) = randAS(DEFAULT_ELTYPE, al)

function randN(T::Type, al::NTuple{N,SBool}) where {N}
    dims = testdims(N)
    innersz = Tuple(dims[i] for i in 1:length(al) if unstatic(al[i]))
    outersz = Tuple(dims[i] for i in 1:length(al) if !unstatic(al[i]))
    return randN(T, innersz, outersz)
end
randN(al::TupleN{SBool}) = randN(DEFAULT_ELTYPE, al)

testdims(L::Integer) = ntuple(i -> 2i, Val(unstatic(L)))
slicedims(al::TupleN{SBool}) = Tuple(static(i) for i=1:N if unstatic(al[i]))

function make_slices(V::Type, alongs::TupleN{SBool})
    L = length(alongs)
    pdims = ntuple(i -> 1 + i, L)
    sdims = Tuple(i for i=1:L if unstatic(alongs[i]))
    innersz = Tuple(pdims[i] for i in 1:L if unstatic(alongs[i]))
    outersz = Tuple(pdims[i] for i in 1:L if !unstatic(alongs[i]))
    M, N = length(innersz), length(outersz)
    flat = rand!(Array{V,L}(undef, pdims...))
    nested = Vector{Array{V,M}}(undef, prod(outersz))
    i = 0
    mapslices(el -> nested[i+=1] = copy(el), flat, dims=sdims)
    Slices(flat, sdims), nested, flat
end

slicedims(al::NTuple{N,SBool}) where {N} = Tuple(static(i) for i=1:N if unstatic(al[i]))

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

    S, nested, flat = make_slices(V, al)
    test_array(() -> Slices(deepcopy(flat), dims), nested)

    continue

    #@testset "functions" begin
    #    S = randS(V, al)

    #    let B = @inferred(flatten(S))
    #        @test B !== parent(S) && B == parent(S)
    #    end
    #    @test @inferred(flatview(S)) === parent(S)

    #    @test @inferred(innersize(S)) == size(first(S))
    #    @test @inferred(inneraxes(S)) == axes(first(S))
    #    @test_noalloc innersize($S)
    #    @test_noalloc inneraxes($S)

    #    @test Slices(parent(S), @inferred(slicedims(S))) === S
    #end

    #@testset "UnsafeArrays" begin
    #    S = randS(V, al)
    #    Sv = uview(S)
    #    @test parent(Sv) isa UnsafeArray{eltype(parent(S)), ndims(parent(S))}
    #    @test S == Sv
    #end

    #@testset "parentindices" begin
    #    S = randS(V, al)
    #    @test all(zip(LinearIndices(S), CartesianIndices(S))) do (i, I)
    #        parentindices(S, i) == parentindices(S, I) == parentindices(S, Tuple(I)...)
    #    end
    #end
end

#@testset "non-standard indexing (AxisArrays)" begin
#    A = AxisArrays.AxisArray(rand(2, 3), [:a, :b], [:x, :y, :z])
#    S = Slices(A, static(true), static(false))
#    @test S[:x] == A[:, :x]
#end
