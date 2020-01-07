@testset "functions" begin
    @testset "inner" begin
        let A = [zeros(2,4) for _=1:6]
            @test @inferred(inner_ndims(A)) == 2

            @test @inferred(inner_size(A)) == (2, 4)
            @test @inferred(inner_size(A, 1)) == 2
            @test @inferred(inner_size(A, 2)) == 4
            @test @inferred(inner_size(A, 3)) == 1

            @test @inferred(inner_axes(A)) == (Base.OneTo(2), Base.OneTo(4))
            @test @inferred(inner_axes(A, 1)) == Base.OneTo(2)
            @test @inferred(inner_axes(A, 2)) == Base.OneTo(4)
            @test @inferred(inner_axes(A, 3)) == Base.OneTo(1)

            @test @inferred(inner_length(A)) == 8
            @test @inferred(inner_eltype(A)) == Float64
        end

        let A = [Int[]]
            @test @inferred(inner_ndims(A)) == 1

            @test @inferred(inner_size(A)) == (0, )
            @test @inferred(inner_size(A, 1)) == 0
            @test @inferred(inner_size(A, 2)) == 1

            @test @inferred(inner_axes(A)) == (Base.OneTo(0), )
            @test @inferred(inner_axes(A, 1)) == Base.OneTo(0)
            @test @inferred(inner_axes(A, 2)) == Base.OneTo(1)

            @test @inferred(inner_length(A)) == 0
            @test @inferred(inner_eltype(A)) == Int
        end

        let A = Vector{Vector{Int}}()
            @test @inferred(inner_ndims(A)) == 1

            @test @inferred(inner_size(A)) == (0, )
            @test @inferred(inner_size(A, 1)) == 0
            @test @inferred(inner_size(A, 2)) == 1

            @test @inferred(inner_axes(A)) == (Base.OneTo(0), )
            @test @inferred(inner_axes(A, 1)) == Base.OneTo(0)
            @test @inferred(inner_axes(A, 2)) == Base.OneTo(1)

            @test @inferred(inner_length(A)) == 0
            @test @inferred(inner_eltype(A)) == Int
        end

        let A = [zeros(2), zeros(4, 6)]
            @test_throws DimensionMismatch @inferred(inner_ndims(A))
            @test_throws DimensionMismatch @inferred(inner_size(A))
            @test_throws DimensionMismatch @inferred(inner_axes(A))
            @test_throws DimensionMismatch @inferred(inner_length(A))
        end

        let A = [zeros(Int, 2), zeros(Float64, 4)]
            @test inner_eltype(A) == Float64
        end

        let A = [zeros(Int, 2), zeros(Float64, 4, 6)]
            @test inner_eltype(A) == Any
        end
    end

    @testset "flatten/flattento!" begin
        @test let A = [rand(2,4) for _=1:6], B = flatten(A)
            mapreduce(vec, vcat, A) == vec(B)
        end
        @test let A = [rand(2,4) for _=1:6], B = flatten(A)
            size(B) == (2,4,6)
        end
        @test let A = [rand(2,4) for _=1:6], B = rand(2,4,6)
            flattento!(B, A)
            mapreduce(vec, vcat, A) == vec(B)
        end
    end
end
