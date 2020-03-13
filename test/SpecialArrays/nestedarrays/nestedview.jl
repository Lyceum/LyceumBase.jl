const RAND_TEST_SIZES = (2,3,2,4,5)
const TEST_M = 0:5
const TEST_U = (Int, Float64)

function randtest(T::Type, ::Val{M}, ::Val{N}) where {M,N}
    dims = ntuple(i -> 2i, Val(M + N))
    M_dims, N_dims = SpecialArrays.split_tuple(dims, Val(M))
    nested = Array{Array{T, M}, N}(undef, N_dims...)
    for i in eachindex(nested)
        x = rand!(zeros(T, M_dims...))
        nested[i] = x
    end
    flat = reshape(mapreduce(vec, vcat, nested), dims)
    return nested, flat
end

nones(::Val{N}) where {N} = ntuple(_ -> 1, Val(N))

@testset "NestedVector" begin
    @test begin
        A = NestedView{0}(zeros(10))
        resize!(A, length(A) + 10)
        length(A) == 20
    end
    A = @inferred(NestedView{0}(zeros(0)))
    xs = Array{Float64,0}[]
    for i=1:10
        x = zeros()
        x .= i
        push!(A, x)
        push!(xs, x)
    end
    for i=1:10
        @test A[i] == xs[i]
    end
end

@testset "NestedView M = $M, N=$N" for M in 0:3, N in 0:3, U in (Float64, Int)
    # TODO https://github.com/JuliaArrays/StaticArrays.jl/issues/705
    if M == N == 0
        @test_skip false
        continue
    end

    L = M + N
    T = let (_, flat) = randtest(U, Val(M), Val(N))
        typeof(view(flat, ncolons(Val(M))..., ntuple(i -> firstindex(flat, M + i), Val(N))...))
    end

    @testset "copyto!" begin
        let
            nested, flat = randtest(U, Val(M), Val(N))
            dest = NestedView{M}(similar(flat))
            src = NestedView{M}(flat)
            @assert dest != src
            copyto!(dest, src)
            @test dest == src
        end
        let
            nested, flat = randtest(U, Val(M), Val(N))
            src = NestedView{M}(flat)
            copyto!(nested, src)
            @test all(zip(nested, src)) do (x, y)
                x == y
            end
        end
        let
            nested, flat = randtest(U, Val(M), Val(N))
            dest = NestedView{M}(flat)
            copyto!(dest, nested)
            @test all(zip(dest, nested)) do (x, y)
                x == y
            end
        end
    end

    continue

    @testset "misc array interface" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))

        @test @inferred(size(A)) == size(flat)[(M+1):(M+N)]
        @test @inferred(axes(A)) == axes(flat)[(M+1):(M+N)]
        @test @inferred(length(A)) == prod(size(A))
        @test @inferred(eltype(A)) === T
        @test @inferred(ndims(A)) == N
        @test @inferred(parent(A)) === parent(A.slices)

        if N > 0
            @test size(@inferred(reshape(A, Val(1)))) == (length(A), )
        end

        if M != N
            @test A != NestedView{L-M}(copy(flat))
        end
    end

    @testset "getindex/setindex!" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))

        @test IndexStyle(A) === IndexStyle(A.slices)
        let I = nones(Val(N)), x = getindex(flat, ncolons(Val(M))..., I...)
            @test _maybe_unsqueeze(@inferred(getindex(A, I...))) == x
        end
        let B = @inferred(getindex(A, :))
            @test size(B) == (length(A), )
            @test parent(B) !== parent(A)
            @test vec(parent(B)) == vec(parent(A))
        end
    end

    if N > 0
        @testset "resize!" begin
            _, flat = randtest(U, Val(M), Val(N))
            A = @inferred(NestedView{M}(ElasticArray(flat)))
            dims, lastdim = Base.front(size(A)), last(size(A))
            resize!(A, dims..., lastdim + 1)
            @test Base.front(size(A)) == dims && last(size(A)) == lastdim + 1
        end
    end

    @testset "copy/deepcopy" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))

        @test A == NestedView{M}(copy(flat))

        B = deepcopy(A)
        @test all(eachindex(A)) do I
            B[I] == A[I]
        end

        for I=eachindex(A)
            x = similar(A[I])
            rand!(x)
            setindex!(A, x, Tuple(I)...)
        end
        @test all(eachindex(A)) do I
            B[I] != A[I]
        end
    end

    @testset "functions" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = NestedView{M}(flat)
        @test flatview(A) === flat
        @test innereltype(typeof(A)) == eltype(typeof(flat)) == U
        @test innersize(A) == size(flat)[1:M]
        @test inneraxes(A) == axes(flat)[1:M]
        @test innerndims(typeof(A)) == M
        @test innerlength(A) == prod(size(flat)[1:M])
        @test innerndims(innerview(flat, Val(M))) == ndims(outerview(flat, Val(M)))

        let B = nestedview(flat, M)
            @test A == B && typeof(A) === typeof(B)
        end
        let B = nestedview(flat, N, false)
            @test A == B && typeof(A) === typeof(B)
        end
    end
end

@testset "parameter checks" begin
    @test_throws ArgumentError check_nestedarray_parameters(Val(1.0),Array{Int,1})
    @test_throws ArgumentError check_nestedarray_parameters(Val(-1),Array{Int,1})
    @test_throws ArgumentError check_nestedarray_parameters(Val(2),Array{Int,1})
end