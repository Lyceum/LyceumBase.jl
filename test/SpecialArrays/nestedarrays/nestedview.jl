const RAND_TEST_SIZES = (2,3,2,4,5)
const TEST_M = 0:5
const TEST_U = (Int, Float64)

@testset "NestedView M = $M, N=$N" for M in 0:3, N in 0:3, U in (Float64, Int)
    # TODO https://github.com/JuliaArrays/StaticArrays.jl/issues/705
    if M == N == 0
        @test_skip false
        continue
    end

    L = M + N
    T = let (_, flat) = randNA(U, static(M), static(N))
        typeof(view(flat, ncolons(M)..., ntuple(i -> firstindex(flat, M + i), Val(N))...))
    end

    @testset "misc array interface" begin
        _, flat = randNA(U, static(M), static(N))
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
        let
            _, flat = randNA(U, static(M), static(N))
            A = @inferred(NestedView{M}(flat))
            I = nones(Val(N))
            @test IndexStyle(A) === IndexStyle(A.slices)
            @test _maybe_unsqueeze(@inferred(getindex(A, I...))) == getindex(flat, ncolons(M)..., I...)
            B = @inferred(getindex(A, :))
            @test size(B) == (length(A), )
            @test parent(B) !== parent(A)
            @test vec(parent(B)) == vec(parent(A))
        end
        let
            _, flat = randNA(U, static(M), static(N))
            A = NestedView{M}(flat)
            x = Array(first(A))
            rand!(x)
            @test first(A) != x && @inferred(setindex!(A, x, 1)) === A && first(A) == x
        end
    end

    @testset "copyto!" begin
        let
            _, flat1 = randNA(U, static(M), static(N))
            _, flat2 = randNA(U, static(M), static(N))
            dest = NestedView{M}(flat1)
            src = NestedView{M}(flat2)
            @test copyto!(dest, src) === dest
            @test dest == src
        end
        let
            dest, _ = randNA(U, static(M), static(N))
            _, flat = randNA(U, static(M), static(N))
            src = NestedView{M}(flat)
            @test copyto!(dest, src) === dest
            @test dest == src
        end
        let
            src, _ = randNA(U, static(M), static(N))
            _, flat = randNA(U, static(M), static(N))
            dest = NestedView{M}(flat)
            @test copyto!(dest, src) === dest
            @test dest == src
        end
    end

    @testset "similar" begin
        _, flat = randNA(U, static(M), static(N))
        A = @inferred(NestedView{M}(flat))
        B = similar(A)
        @test typeof(A) == typeof(B) && axes(A) == axes(B)
    end

    if N > 0
        @testset "resize!" begin
            _, flat = randNA(U, static(M), static(N))
            A = @inferred(NestedView{M}(ElasticArray(flat)))
            dims, lastdim = Base.front(size(A)), last(size(A))
            resize!(A, dims..., lastdim + 1)
            @test Base.front(size(A)) == dims && last(size(A)) == lastdim + 1
        end
    end

    @testset "functions" begin
        _, flat = randNA(U, static(M), static(N))
        A = NestedView{M}(flat)
        @test flatview(A) === flat
        @test inneraxes(A) == axes(flat)[1:M]
        @test innersize(A) == size(flat)[1:M]

        @test @inferred(nestedview(flat, static(M))) isa NestedView{M,<:Any,N}
        @test @inferred(nestedview(flat, static(M), inner = static(false))) isa NestedView{N,<:Any,M}
    end
end

@testset "similar" begin
    M = 2
    N = 3
    A = NestedView{M}(last(randNA(Float64, M, N)))
    @test_throws ArgumentError similar(A, Array{Float64, M+1})
    let B = similar(A, Array{Int,M})
        @test size(B) == size(A) && typeof(B) <: NestedView{M,<:AbsArr{Int,M}, N}
    end
    let newdims = (map(d -> d + 1, size(A))..., 10), B = similar(A, Array{Int,M}, newdims)
        @test size(B) == newdims && typeof(B) <: NestedView{M,<:AbsArr{Int,M}, N + 1}
    end
end

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

@testset "parameter checks" begin
    @test_throws ArgumentError check_nestedarray_parameters(Val(1.0),Array{Int,1})
    @test_throws ArgumentError check_nestedarray_parameters(Val(-1),Array{Int,1})
    @test_throws ArgumentError check_nestedarray_parameters(Val(2),Array{Int,1})
end