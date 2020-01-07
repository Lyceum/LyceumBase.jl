const TEST_DIMS = (2, 4, 6)
const TEST_KERNEL_SIZE = Base.front(TEST_DIMS)
const TEST_KERNEL_LENGTH = Base.MultiplicativeInverses.SignedMultiplicativeInverse(prod(TEST_KERNEL_SIZE))
const T = Float64
const N = length(TEST_DIMS)
const M = N - 1

function test_A()
    A = Array{T}(undef, TEST_DIMS...)
    copyto!(A, 1:prod(TEST_DIMS))
end

function test_V()
    V = Vector{T}(undef, prod(TEST_DIMS))
    copyto!(V, 1:prod(TEST_DIMS))
end

function test_E()
    V = test_V()
    ElasticBuffer{T}(TEST_KERNEL_SIZE, V)
end

lastdim_slice_idxs(A::AbstractArray{T,N}, i::Integer) where {T,N} = (ntuple(_ -> :, Val(N - 1))..., i)

test_comp(E::ElasticBuffer, V::Vector{<:Array}) =
    all(i -> @view(E[lastdim_slice_idxs(E, i)...]) == V[i], eachindex(V))

@testset "constructors" begin
    let V = test_V()
        @test @inferred(ElasticBuffer{T,N,M}(TEST_KERNEL_SIZE, V)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T,N,M}(TEST_KERNEL_SIZE, V).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T,N,M}(TEST_KERNEL_SIZE, V).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T,N,M}(TEST_KERNEL_SIZE, V).data === V

        @test @inferred(ElasticBuffer{T,N}(TEST_KERNEL_SIZE, V)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T,N}(TEST_KERNEL_SIZE, V).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T,N}(TEST_KERNEL_SIZE, V).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T,N}(TEST_KERNEL_SIZE, V).data === V

        @test @inferred(ElasticBuffer{T}(TEST_KERNEL_SIZE, V)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T}(TEST_KERNEL_SIZE, V).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T}(TEST_KERNEL_SIZE, V).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T}(TEST_KERNEL_SIZE, V).data === V
    end


    @test @inferred(ElasticBuffer{T,N,M}(undef, TEST_DIMS)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T,N,M}(undef, TEST_DIMS).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T,N,M}(undef, TEST_DIMS).kernel_length == TEST_KERNEL_LENGTH

    @test @inferred(ElasticBuffer{T,N}(undef, TEST_DIMS)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T,N}(undef, TEST_DIMS).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T,N}(undef, TEST_DIMS).kernel_length == TEST_KERNEL_LENGTH

    @test @inferred(ElasticBuffer{T}(undef, TEST_DIMS)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T}(undef, TEST_DIMS).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T}(undef, TEST_DIMS).kernel_length == TEST_KERNEL_LENGTH


    @test @inferred(ElasticBuffer{T,N,M}(undef, TEST_DIMS...)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T,N,M}(undef, TEST_DIMS...).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T,N,M}(undef, TEST_DIMS...).kernel_length == TEST_KERNEL_LENGTH

    @test @inferred(ElasticBuffer{T,N}(undef, TEST_DIMS...)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T,N}(undef, TEST_DIMS...).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T,N}(undef, TEST_DIMS...).kernel_length == TEST_KERNEL_LENGTH

    @test @inferred(ElasticBuffer{T}(undef, TEST_DIMS...)) isa ElasticBuffer{T,N,M}
    @test ElasticBuffer{T}(undef, TEST_DIMS...).kernel_size == TEST_KERNEL_SIZE
    @test ElasticBuffer{T}(undef, TEST_DIMS...).kernel_length == TEST_KERNEL_LENGTH


    let A = test_A()
        @test @inferred(ElasticBuffer{T,N,M}(A)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T,N,M}(A).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T,N,M}(A).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T,N,M}(A).data == vec(A)
        @test ElasticBuffer{T,N,M}(A).data !== A

        @test @inferred(ElasticBuffer{T,N}(A)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T,N}(A).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T,N}(A).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T,N}(A).data == vec(A)
        @test ElasticBuffer{T,N}(A).data !== A

        @test @inferred(ElasticBuffer{T}(A)) isa ElasticBuffer{T,N,M}
        @test ElasticBuffer{T}(A).kernel_size == TEST_KERNEL_SIZE
        @test ElasticBuffer{T}(A).kernel_length == TEST_KERNEL_LENGTH
        @test ElasticBuffer{T}(A).data == vec(A)
        @test ElasticBuffer{T}(A).data !== A
    end
end

@testset "convert" begin
    A = test_A()

    @test @inferred(convert(ElasticBuffer{T,N,M}, A)) isa ElasticBuffer{T,N,M}
    @test convert(ElasticBuffer{T,N,M}, A).kernel_size == TEST_KERNEL_SIZE
    @test convert(ElasticBuffer{T,N,M}, A).kernel_length == TEST_KERNEL_LENGTH
    @test convert(ElasticBuffer{T,N,M}, A).data == vec(A)
    @test convert(ElasticBuffer{T,N,M}, A).data !== A

    @test @inferred(convert(ElasticBuffer{T,N}, A)) isa ElasticBuffer{T,N,M}
    @test convert(ElasticBuffer{T,N}, A).kernel_size == TEST_KERNEL_SIZE
    @test convert(ElasticBuffer{T,N}, A).kernel_length == TEST_KERNEL_LENGTH
    @test convert(ElasticBuffer{T,N}, A).data == vec(A)
    @test convert(ElasticBuffer{T,N}, A).data !== A

    @test @inferred(convert(ElasticBuffer{T}, A)) isa ElasticBuffer{T,N,M}
    @test convert(ElasticBuffer{T}, A).kernel_size == TEST_KERNEL_SIZE
    @test convert(ElasticBuffer{T}, A).kernel_length == TEST_KERNEL_LENGTH
    @test convert(ElasticBuffer{T}, A).data == vec(A)
    @test convert(ElasticBuffer{T}, A).data !== A
end

@testset "misc array interface" begin
    E, A = test_E(), test_A()
    @test @inferred(size(E)) == size(A)
    @test @inferred(axes(E)) == axes(A)
    for d = 1:length(TEST_DIMS)
        @test @inferred(size(E, d)) == size(A, d)
        @test @inferred(axes(E, d)) == axes(A, d)
    end
    @test @inferred(length(E)) == length(A)
    @test @inferred(eltype(E)) == eltype(A)
    @test @inferred(ndims(E)) == ndims(A)
    @test IndexStyle(E) == IndexLinear()
    @test Base.dataids(E) == Base.dataids(E.data)
end

@testset "getindex/setindex!" begin
    E, A = test_E(), test_A()
    rand!(A)
    for i in eachindex(E, A)
        x = A[i]
        @inferred setindex!(E, x, i)
    end
    @test all(eachindex(E, A)) do i
        @inferred(getindex(E, i)) == A[i]
    end
end

@testset "equality" begin
    let E = test_E(), A = test_A()
        @test E == @inferred(deepcopy(E))
        @test @inferred(E == A)
    end
end

@testset "mightalias and dataids" begin
    E1 = ElasticBuffer{T}(undef, 10, 5)
    E2 = ElasticBuffer{T}(undef, 10, 5)
    @test Base.dataids(parent(E1)) == @inferred Base.dataids(E1)
    @test @inferred !Base.mightalias(E1, E2)
    @test @inferred !Base.mightalias(view(E1, 2:3, 1:2), view(E1, 4:5, 1:2))
    @test @inferred Base.mightalias(view(E1, 2:4, 1:2), view(E1, 3:5, 1:2))
    @test @inferred !Base.mightalias(view(E1, 2:4, 1:2), view(E2, 3:5, 1:2))
end

@testset "similar" begin
    E = test_E()
    @test typeof(@inferred(similar(E))) === Array{T,N}
    @test size(similar(E)) == size(E)

    @test typeof(@inferred(similar(E, Int))) === Array{Int,N}
    @test size(similar(E, Int)) == size(E)
    @test eltype(similar(E, Int)) != eltype(E)

    dims = (TEST_DIMS..., 10)
    @test typeof(@inferred(similar(E, T, dims))) === Array{T,N+1}
    @test size(similar(E, T, dims)) == dims
    @test eltype(similar(E, T, dims)) == eltype(E)
end


@testset "copyto!" begin
    let E = test_E(), A = rand!(test_A())
        @test E != A
        @test E === @inferred copyto!(E, A)
        @test E == A
    end

    let E = rand!(test_E()), A = test_A()
        @test E != A
        copyto!(A, E)
        @test E == A
    end

    let E = test_E(), A = rand!(test_A())
        @test E != A
        @test E === @inferred copyto!(E, 2, A, 3, 5)
        @test E[1] != A[1]
        @test E[2:6] == A[3:7]
        @test E[7:end] != A[8:7]
    end

    let E = rand!(test_E()), A = test_A()
        @test E != A
        copyto!(A, 2, E, 3, 5)
        @test A[1] != E[1]
        @test A[2:6] == E[3:7]
        @test A[7:end] != E[8:end]
    end
end

@testset "pointer and unsafe_convert" begin
    let E = test_E()
        @test pointer(E) == pointer(parent(E))
        @test pointer(E, 4) == pointer(parent(E), 4)
    end
    let E = test_E()
        @test Base.unsafe_convert(Ptr{eltype(E)}, E) == Base.unsafe_convert(Ptr{eltype(E)}, parent(E))
    end
end

@testset "resize!" begin
    function resize_test(delta::Integer)
        let E = test_E()
            A = Array(deepcopy(E))
            new_size = (Base.front(size(E))..., size(E, ndims(E)) + delta)
            cmp_idxs = (Base.front(axes(E))..., 1:(last(size(E)) + min(0, delta)))
            @test E === @inferred sizehint!(E, new_size...)
            @test E === @inferred resize!(E, new_size...)
            @test size(E) == new_size
            @test E[cmp_idxs...] == A[cmp_idxs...]
        end
    end

    resize_test(0)
    resize_test(2)
    resize_test(-2)
end

@testset "append! and prepend!" begin
    let E = test_E()
        dims = Base.front(size(E))
        len_lastdim = size(E, ndims(E))
        V = Array{T, length(dims)}[]
        for i in 1:4
            push!(V, rand(dims...))
            @inferred append!(E, last(V))
        end
        @test size(E) == (dims..., len_lastdim + length(V))
        @test all(1:length(V)) do i
            selectdim(E, ndims(E), i + len_lastdim) == V[i]
        end
    end

    let E = test_E()
        dims = Base.front(size(E))
        len_lastdim = size(E, ndims(E))
        V = Array{T, length(dims)}[]
        for i in 1:4
            pushfirst!(V, rand(dims...))
            @inferred prepend!(E, first(V))
        end
        @test size(E) == (dims..., len_lastdim + length(V))
        @test all(1:length(V)) do i
            selectdim(E, ndims(E), i) == V[i]
        end
    end
end

@testset "basic math" begin
    E1 = rand!(ElasticBuffer{T}(undef, 9, 9))
    E2 = rand!(ElasticBuffer{T}(undef, 9, 9))
    E3 = rand!(ElasticBuffer{T}(undef, 9, 7))

    A1 = Array(E1)
    A2 = Array(E2)
    A3 = Array(E3)

    @test @inferred(2 * E1) isa Array{T,2}
    @test 2 * E1 == 2 * A1

    @test @inferred(E1 .+ 2) isa Array{T,2}
    @test E1 .+ 2 == A1 .+ 2

    @test @inferred(E1 + E2) isa Array{T,2}
    @test E1 + E2 == A1 + A2

    @test @inferred(E1 * E2) isa Array{T,2}
    @test E1 * E2 == A1 * A2
    @test E1 * E3 == A1 * A3

    @test E1^3 == A1^3
    @test inv(E1) == inv(A1)
end

@testset "growlastdim!, shrinklastdim!, resizelastdim!" begin
    let E = test_E()
        dims, d = Base.front(size(E)), last(size(E))
        growlastdim!(E, 2)
        @test size(E) == (dims..., d + 2)
    end

    let E = test_E()
        dims, d = Base.front(size(E)), last(size(E))
        shrinklastdim!(E, 2)
        @test size(E) == (dims..., d - 2)
    end

    let E = test_E()
        dims, d = Base.front(size(E)), last(size(E))
        resizelastdim!(E, 2)
        @test size(E) == (dims..., 2)
    end
end