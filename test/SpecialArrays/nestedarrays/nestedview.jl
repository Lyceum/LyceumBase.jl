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
        A = NestedVector{0}(zeros(10))
        resize!(A, length(A) + 10)
        length(A) == 20
    end
    let
        A = @inferred(NestedVector{0}(zeros(0)))
        B = @inferred(NestedVector(zeros(0)))
        C = NestedVector{0}(zeros(0))
        xs = Array{Float64,0}[]
        for i=1:10
            x = zeros()
            x .= i
            push!(A, x)
            push!(xs, x)
        end
        append!(B, xs)
        append!(C, B)
        for i=1:10
            @test A[i] == xs[i]
            @test B[i] == xs[i]
            @test C[i] == xs[i]
        end
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

    @testset "constructors" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = NestedView{M}(flat)
        AT = NestedView{M,T,N,Array{U,M+N}}

        @test A == @inferred(convert(NestedView{M,T,N,Array{U,M+N}}, flat))
        @test A == @inferred(convert(NestedView{M,T,N}, flat))
        @test A == @inferred(convert(NestedView{M,T}, flat))
        @test A == @inferred(convert(NestedView{M}, flat))

        @test typeof(convert(NestedView{M,T,N,Array{U,M+N}}, flat)) <: AT
        @test typeof(convert(NestedView{M,T,N}, flat)) <: AT
        @test typeof(convert(NestedView{M,T}, flat)) <: AT
        @test typeof(convert(NestedView{M}, flat)) <: AT

        @test A == @inferred(NestedView{M,T,N,Array{U,M+N}}(flat))
        @test A == @inferred(NestedView{M,T,N}(flat))
        @test A == @inferred(NestedView{M,T}(flat))
        @test A == @inferred(NestedView{M}(flat))
        @test A == @inferred(NestedView{M}(flat))

        @test typeof(NestedView{M,T,N,Array{U,M+N}}(flat)) <: AT
        @test typeof(NestedView{M,T,N}(flat)) <: AT
        @test typeof(NestedView{M,T}(flat)) <: AT
        @test typeof(NestedView{M}(flat)) <: AT

        @test A == @inferred(innerview(flat, Val(M)))
        @test typeof(innerview(flat, Val(M))) <: AT
        @test typeof(innerview(flat, M)) <: AT

        @test typeof(@inferred(similar(A))) === typeof(A)
        @test typeof(@inferred(similar(A, Array{U,N}))) <: NestedView{N,<:AbstractArray{U,N},M,Array{U,M+N}}
    end

    @testset "parameter checks" begin
        @test_throws ArgumentError check_nestedarray_parameters(Val(M+1),T,Val(N),Array{U,M+N})
        @test_throws ArgumentError check_nestedarray_parameters(Val(M),T,Val(N+1),Array{U,M+N})
        @test_throws ArgumentError check_nestedarray_parameters(Val(M),T,Val(N),Array{U,M+N+1})
        @test_throws DomainError check_nestedarray_parameters(Val(-1),T,Val(N),Array{U,M+N})
        @test_throws DomainError check_nestedarray_parameters(Val(M),T,Val(-1),Array{U,M+N})
    end

    @testset "misc array interface" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))
        @test @inferred(size(A)) == size(flat)[(M+1):(M+N)]
        @test @inferred(axes(A)) == axes(flat)[(M+1):(M+N)]
        @test @inferred(length(A)) == prod(size(A))
        @test @inferred(eltype(A)) === T
        @test @inferred(ndims(A)) == N
        if N > 0
            @test size(@inferred(reshape(A, Val(1)))) == (length(A), )
        end
        @test A == NestedView{M}(copy(flat))
        @test parent(A) === A.parent
        if M != N
            @test A != NestedView{L-M}(copy(flat))
        end
        let B = similar(A, size(A)..., 5)
            @test ndims(B) == ndims(A) + 1
            @test size(B) == (size(A)..., 5)
        end
        let B = NestedView{M}(similar(flat))
            @assert B.parent !== flat
            A.parent .= 0
            B.parent .= 1
            copyto!(B, A)
            @test B == A
        end
    end

    @testset "getindex/setindex!" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))

        @test IndexStyle(A) === IndexStyle(A.parent)
        let I = nones(Val(N)), x = getindex(flat, ncolons(Val(M))..., I...)
            @test _maybe_unsqueeze(@inferred(getindex(A, I...))) == x
        end
        let B = @inferred(getindex(A, :))
            @test size(B) == (length(A), )
            @test parent(B) !== parent(A)
            @test vec(parent(B)) == vec(parent(A))
        end
    end

    @testset "deepcopy" begin
        _, flat = randtest(U, Val(M), Val(N))
        A = @inferred(NestedView{M}(flat))
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
        @test inner_eltype(typeof(A)) == eltype(typeof(flat)) == U
        @test inner_size(A) == size(flat)[1:M]
        @test inner_axes(A) == axes(flat)[1:M]
        @test inner_ndims(typeof(A)) == M
        @test inner_length(A) == prod(size(flat)[1:M])
        @test inner_ndims(innerview(flat, Val(M))) == ndims(outerview(flat, Val(M)))
    end
end