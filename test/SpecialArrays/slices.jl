const TEST_ALONGS = [
    (static(true), ),
    (static(false), ),

    (static(true), static(true)),
    (static(false), static(true)),
    (static(false), static(false)),
]

slicedims(al::TupleN{SBool}) = Tuple(i for i=1:length(al) if unstatic(al[i]))

# TODO add Show for slices
function _show_alongs(io, alongs::TupleN{SBool})
    write(io, '(')
    if length(alongs) == 1
        write(io, unstatic(alongs[1]) ? ':' : "i1", ',')
    elseif length(alongs) > 1
        write(io, unstatic(alongs[1]) ? ':' : "i1")
        for dim = 2:length(alongs)
            write(io, ',', ' ', unstatic(alongs[dim]) ? ':' : "i$dim")
        end
    end
    write(io, ')')
    return nothing
end

function _show_alongs(alongs::TupleN{SBool})
    io = IOBuffer()
    _show_alongs(io, alongs)
    String(take!(io))
end


@testset "Slices $V $(_show_alongs(al))" for V in (Float64, ), al in TEST_ALONGS
    function test_SNF()
        L = length(al)
        pdims = testdims(L)
        sdims = slicedims(al)
        innersz = Tuple(pdims[i] for i in 1:L if unstatic(al[i]))
        outersz = Tuple(pdims[i] for i in 1:L if !unstatic(al[i]))
        M, N = length(innersz), length(outersz)
        flat = rand!(Array{V,L}(undef, pdims...))
        nested = Array{Array{V,M},N}(undef, outersz...)
        i = 0
        Base.mapslices(flat, dims=sdims) do el
            i += 1
            nested[i] = zeros(V, innersz...)
            nested[i] .= el
            el
        end
        Slices(flat, al), nested, flat
    end

    @testset "constructors" begin
        _, nested, flat = test_SNF()
        sdims = slicedims(al)
        static_sdims = map(static, sdims)
        M = ndims(first(nested))
        N = ndims(nested)
        ST = Slices{<:AbsArr{V,M},N,M,Array{V,M+N},typeof(al)}

        @test typeof(Slices(flat, al)) <: ST
        @test_inferred Slices(flat, al)

        @test typeof(slice(flat, al)) <: ST
        @test_inferred slice(flat, al)
        @test typeof(slice(flat, al...)) <: ST
        @test_inferred slice(flat, al...)

        @test_inferred slice(flat, static_sdims)
        @test typeof(slice(flat, static_sdims)) <: ST
        @test_inferred slice(flat, static_sdims...)
        @test typeof(slice(flat, static_sdims...)) <: ST

        @test typeof(slice(flat, sdims)) <: ST
        @test typeof(slice(flat, sdims...)) <: ST
    end

    @testset "array attributes" begin
        S, _, _ = test_SNF()
        @test_array_attributes S

        @test_noalloc eltype($S)
        @test_noalloc ndims($S)
        @test_noalloc axes($S)
        @test_noalloc size($S)
        @test_noalloc length($S)
    end

    @time begin
    @testset "indexing" begin
        S, nested, flat =  test_SNF()
        test_indexing_AB(() -> Slices(deepcopy(flat), al), nested)
    end
end

    @testset "misc" begin
        S, _, _ = test_SNF()
        @test parent(S) === S.parent
        @test Base.dataids(S) === Base.dataids(S.parent)
    end

    @testset "copy/copyto!/equality" begin
        let (S, nested, _) = test_SNF()
            @test_copyto! S nested
        end
        let (S, nested, _) = test_SNF()
            @test_copyto! nested S
        end
        let S1 = first(test_SNF()), S2 = first(test_SNF())
            @test_copyto! S1 S2
        end
    end

    @testset "Extra" begin
        S, nested, _ = test_SNF()

        @test begin
            flat = flatten(S)
            flat !== S.parent && flat == S.parent
        end
        @test_inferred flatten(S)

        @test flatview(S) === S.parent
        @test_inferred flatview(S)

        @test innersize(S) == size(first(S))
        @test_inferred innersize(S)
        @test_noalloc inneraxes($S)

        @test inneraxes(S) == axes(first(S))
        @test_inferred inneraxes(S)
        @test_noalloc innersize($S)
    end

    @testset "UnsafeArrays" begin
        S, _, _= test_SNF()
        Sv = uview(S)
        @test parent(Sv) isa UnsafeArray{eltype(S.parent), ndims(S.parent)}
        @test S == Sv
    end

    @testset "mapslicea" for f in (
        identity,
        el -> sum(el),
        el -> el isa AbsArr ? reshape(el, reverse(size(el))) : el,
        el -> el isa AbsArr ? reshape(el, Val(1)) : el,
    )
        # dropdims=false/Base.mapslices behavior
        let (S, _, flat) = test_SNF()
            @test_inferred SpecialArrays.mapslices(f, flat, dims=al)
            B1 = mapslices(f, flat, dims=slicedims(al))
            B2 = flatview(SpecialArrays.mapslices(f, flat, dims=al))
            @test B1 == B2
        end

        # dropdims=true
        let (S, _, flat) = test_SNF()
            @test_inferred SpecialArrays.mapslices(f, flat, dims=al, dropdims=static(true))
            B1 = map(f, slice(flat, slicedims(al)))
            B2 = SpecialArrays.mapslices(f, flat, dims=al, dropdims=static(true))
            @test B1 == B2
        end
    end
end

#@testset "non-standard indexing (AxisArrays)" begin
#    A = AxisArrays.AxisArray(rand(2, 3), [:a, :b], [:x, :y, :z])
#    S = Slices(A, static(true), static(false))
#    @test S[:x] == A[:, :x]
#end
