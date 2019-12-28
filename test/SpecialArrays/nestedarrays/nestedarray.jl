const RAND_TEST_SIZES = (2,3,2,4,5)
const TEST_M = 0:5
const TEST_U = (Int, Float64)

randflat(T::Type, ::Val{0}) = (x = zeros(T); x[] = rand(T); x)
function randflat(T::Type, ::Val{L}) where {L}
    if L == 0
        x = zeros(T)
        x[] = rand(T)
        return x
    else
        sz = ntuple(i -> 2*i, Val(L))
        x = rand(T, sz...)
    end
end

function randnested(T::Type, ::Val{M}, ::Val{N}) where {M,N}
    sz_inner = ntuple(i -> RAND_TEST_SIZES[i], M)
    sz_outer = ntuple(i -> RAND_TEST_SIZES[i + M], N)
    A = Array{Array{T, M}, N}(undef, sz_outer...)
    for i in eachindex(A)
        A[i] = rand(sz_inner...)
    end
    A
end

nones(::Val{N}) where {N} = ntuple(_ -> 1, Val(N))

@generated function testindices(::Val{M}, ::Val{N}) where {M,N}
    ex = Expr(:tuple)
    I = ncolons(Val(N))
    push!(ex.args, (I, (ncolons(Val(M))..., I...)))

    I = nones(Val(N))
    push!(ex.args, (I, (ncolons(Val(M))..., I...)))

    if M > 1
        I = (ncolons(Val(N-1))..., 1)
        push!(ex.args, (I, (ncolons(Val(M))..., I...)))
    end
    ex
end

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
    T = SpecialArrays.viewtype(randflat(U, Val(M+N)), Val(M), Val(N))
    L = M + N

    @testset "constructors" begin
        flat = randflat(U, Val(M+N))
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

    @testset "basic array interface" begin
        flat = randflat(U, Val(M+N))
        A = @inferred(NestedView{M}(flat))
        @test @inferred(size(A)) == size(flat)[(M+1):(M+N)]
        @test @inferred(eltype(A)) === T
        @test @inferred(ndims(A)) == N
        @test @inferred(length(inner_size(A))) == M
        @test A == NestedView{M}(copy(flat))

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

    @testset "functions" begin
        flat = randflat(U, Val(M+N))
        A = NestedView{M}(flat)
        @test flatview(A) === flat
        @test inner_eltype(typeof(A)) == eltype(typeof(flat)) == U
        @test inner_size(A) == size(flat)[1:M]
        @test inner_axes(A) == axes(flat)[1:M]
        @test inner_ndims(typeof(A)) == M
        @test inner_length(A) == prod(size(flat)[1:M])
        @test inner_ndims(innerview(flat, Val(M))) == ndims(outerview(flat, Val(M)))
    end

    @testset "getindex/setindex!" begin
        flat = randflat(U, Val(M+N))
       A = @inferred(NestedView{M}(flat))

        I = nones(Val(N))
        let x = getindex(flat, ncolons(Val(M))..., I...)
            @test _maybe_unsqueeze(@inferred(getindex(A, I...))) == x
        end
        let B = @inferred(getindex(A, :))
            @test size(B) == (length(A), )
            @test parent(B) !== parent(A)
            @test vec(parent(B)) == vec(parent(A))
        end
    end

    @testset "deepcopy" begin
        flat = randflat(U, Val(M+N))
        A = @inferred(NestedView{M}(flat))
        B = deepcopy(A)
        @test all(eachindex(A)) do I
            B[I] == A[I]
        end
        for I=eachindex(A)
            x = similar(A[I])
            x .= 1
            #if x isa AbstractArray{T, 0} where T
            #    x = x[]
            #end
            setindex!(A, x, Tuple(I)...)
        end
        @test all(eachindex(A)) do I
            B[I] != A[I]
        end
    end
end