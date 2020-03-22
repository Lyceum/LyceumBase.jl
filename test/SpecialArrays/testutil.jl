randlike(x::Number) = rand(typeof(x))
randlike(A::AbstractArray{<:Number}) = rand(eltype(A), size(A)...)
randlike(A::AbstractArray{<:Number,0}) = rand!(zeros(eltype(A)))
function randlike(A::AbsArr{<:AbsArr{V,M},N}) where {V,M,N}
    B = Array{Array{V,M},N}(undef, size(A)...)
    for I in eachindex(B)
        B[I] = randlike(A[I])
    end
    return B
end
function randlike!(A::AbsArr{<:AbsArr})
    for I in eachindex(A)
        copyto!(A[I], randlike(A[I]))
    end
    return A
end
function randlike!(A::AbsArr{<:Number})
    for I in eachindex(A)
        A[I] = randlike(A[I])
    end
    return A
end

function test_indices(A::AbsArr, B::Array, Is::AbsArr{<:Tuple})
    I1 = first(Is)

    @test_inferred A[I1...]
    @test A[I1...] == B[I1...]
    @test compare_type_and_shape(A, B, I1...)
    @test all(Is) do I
        A[I...] == B[I...] && compare_type_and_shape(A, B, I...)
    end

    x = randlike(B[I1...])
    @test_inferred setindex!(A, x, I1...)
    @test setindex!(A, x, I1...) === A
    @test A[I1...] == x
    @test all(Is) do I
        B[I...] = randlike(B[I...])
        @assert A[I...] != B[I...]
        A[I...] = B[I...]
        A[I...] == B[I...] && compare_type_and_shape(A, B, I...)
    end
end



function test_indices(A::AbsArr, B::Array, Is::AbsArr)
    I1 = first(Is)

    @test_inferred A[I1]
    @test A[I1] == B[I1]
    @test compare_type_and_shape(A, B, I1)
    @test all(Is) do I
        A[I] == B[I] && compare_type_and_shape(A, B, I)
    end

    x = randlike(B[I1])
    @test_inferred setindex!(A, x, I1)
    @test setindex!(A, x, I1) === A
    @test A[I1] == x
    @test all(Is) do I
        B[I] = randlike(B[I])
        @assert A[I] != B[I]
        A[I] = B[I]
        A[I] == B[I] && compare_type_and_shape(A, B, I)
    end
end

function compare_type_and_shape(A::AbsArr, B::Array, I...)
    a, b = A[I...], B[I...]
    J = Base.to_indices(A, I)
    if Base.index_dimsum(J...) === ()
        return typeof(a) === eltype(A)
    else
        return axes(a) == Base.index_shape(J...) && eltype(a) === eltype(A)
    end
end


function test_copyto!(dest::AbsArr, src::AbsArr)
    randlike!(dest)
    @test_inferred copyto!(dest, src)
    @test copyto!(dest, src) === dest
    @test all(I -> dest[I] == src[I], eachindex(dest, src))
    @test dest == src

    if length(dest) > 2
        randlike!(dest)
        @test_inferred copyto!(dest, 2, src, 2, 1)
        @test copyto!(dest, 2, src, 2, 1) === dest
        @test dest[2] == src[2] && dest[1] != src[1] && dest[3:end] != src[3:end]
    end
end

function test_array(f::Function, B0::Array)
    AB = let f=f, B0=B0
        () -> return f(), deepcopy(B0)
    end

    @testset "misc array interface" begin
        A, B = AB()
        #@test @inferred(eltype(A)) === eltype(B)
        @test @inferred(ndims(A)) == length(axes(A)) == ndims(B)
        @test @inferred(axes(A)) == axes(B)
        @test @inferred(size(A)) == map(length, axes(A)) == size(B)
        @test @inferred(length(A)) == prod(size(A)) == length(B)
    end

    @testset "indexing" begin
        @testset "LinearIndices" begin
            A, B = AB()
            @test LinearIndices(A) === LinearIndices(B)
            test_indices(A, B, LinearIndices(A))
        end
        @testset "CartesianIndices" begin
            A, B = AB()
            @test CartesianIndices(A) === CartesianIndices(B)
            test_indices(A, B, CartesianIndices(A))
        end
        @testset "trailing singleton" begin
            A, B = AB()
            test_indices(A, B, map(I -> (Tuple(I)..., 1), CartesianIndices(A)))
        end
        @testset "trailing colon" begin
            A, B = AB()
            test_indices(A, B, map(I -> (Tuple(I)..., :), CartesianIndices(A)))
        end
        @testset "dropped singleton" begin
            A, B = AB()
            if size(B, ndims(B)) == 1
                test_indices(A, B, map(I -> front(Tuple(I)), CartesianIndices(A)))
            end
        end
        @testset "logical" begin
            A, B = AB()
            I = map(isodd, LinearIndices(A))
            @test A[I] == B[I]
        end
        @testset "single colon" begin
            A, B = AB()
            A[:] == B[:]
        end
        @testset "range" begin
            A, B = AB()
            I = Tuple(1:max(1, length(ax) - 1) for ax in axes(A))
            test_indices(A, B, [I])
        end
    end

    @testset "similar" begin
        A, _  = AB()
        if eltype(A) <: Number
            T = eltype(A) == Float64 ? Int : Float64
        elseif eltype(A) <: AbsArr
            V = eltype(eltype(A)) == Float64 ? Int : Float64
            T = AbstractArray{V,ndims(eltype(A)) + 1}
        end
        dims = (size(A)..., 10)

        @test_inferred similar(A)
        let A2 = similar(A)
            @test eltype(A2) === eltype(A)
            @test size(A2) == size(A)
        end
        @test_inferred similar(A, T)
        let A2 = similar(A, T)
            @test eltype(A2) <: T
            @test size(A2) == size(A)
        end
        @test_inferred similar(A, dims)
        let A2 = similar(A, dims)
            @test eltype(A2) == eltype(A)
            @test size(A2) == dims
        end
        @test_inferred similar(A, T, dims)
        let A2 = similar(A, T, dims)
            @test eltype(A2) <: T
            @test size(A2) == dims
        end
    end

    @testset "copy/copyto!/equality" begin
        A, B = AB()
        test_copyto!(A, B)
        test_copyto!(B, A)
        A2, _ = AB()
        test_copyto!(A, A2)
        @test copy(A) == A
    end
end
