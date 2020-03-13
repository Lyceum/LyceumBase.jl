@testset "functions" begin
    @testset "inner" begin
        let A = [zeros(2,4) for _=1:6]
            @test @inferred(innerndims(A)) == 2

            @test @inferred(innersize(A)) == (2, 4)
            @test @inferred(innersize(A, 1)) == 2
            @test @inferred(innersize(A, 2)) == 4
            @test @inferred(innersize(A, 3)) == 1

            @test @inferred(inneraxes(A)) == (Base.OneTo(2), Base.OneTo(4))
            @test @inferred(inneraxes(A, 1)) == Base.OneTo(2)
            @test @inferred(inneraxes(A, 2)) == Base.OneTo(4)
            @test @inferred(inneraxes(A, 3)) == Base.OneTo(1)

            @test @inferred(innerlength(A)) == 8
            @test @inferred(innereltype(A)) == Float64
        end

        let A = [Int[]]
            @test @inferred(innerndims(A)) == 1

            @test @inferred(innersize(A)) == (0, )
            @test @inferred(innersize(A, 1)) == 0
            @test @inferred(innersize(A, 2)) == 1

            @test @inferred(inneraxes(A)) == (Base.OneTo(0), )
            @test @inferred(inneraxes(A, 1)) == Base.OneTo(0)
            @test @inferred(inneraxes(A, 2)) == Base.OneTo(1)

            @test @inferred(innerlength(A)) == 0
            @test @inferred(innereltype(A)) == Int
        end

        let A = Vector{Vector{Int}}()
            @test @inferred(innerndims(A)) == 1
            @test @inferred(innerndims(typeof(A))) == 1

            @test @inferred(innersize(A)) == (0, )
            @test @inferred(innersize(A, 1)) == 0
            @test @inferred(innersize(A, 2)) == 1

            @test @inferred(inneraxes(A)) == (Base.OneTo(0), )
            @test @inferred(inneraxes(A, 1)) == Base.OneTo(0)
            @test @inferred(inneraxes(A, 2)) == Base.OneTo(1)

            @test @inferred(innerlength(A)) == 0
            @test @inferred(innereltype(A)) == Int
            @test @inferred(innereltype(typeof(A))) == Int
        end

        let A = [zeros(2), zeros(4, 6)]
            @test_throws DimensionMismatch @inferred(innerndims(A))
            @test_throws DimensionMismatch @inferred(innersize(A))
            @test_throws DimensionMismatch @inferred(inneraxes(A))
            @test_throws DimensionMismatch @inferred(innerlength(A))
        end

        let A = [zeros(Int, 2), zeros(Float64, 4)]
            @test innereltype(A) == Float64
        end

        let A = [zeros(Int, 2), zeros(Float64, 4, 6)]
            @test innereltype(A) == Any
        end
    end

    @testset "flatten/flatten!" begin
        @test let A = [rand(2,4) for _=1:6], B = flatten(A)
            mapreduce(vec, vcat, A) == vec(B)
        end
        @test let A = [rand(2,4) for _=1:6], B = flatten(A)
            size(B) == (2,4,6)
        end
        @test let A = [rand(2,4) for _=1:6], B = rand(2,4,6)
            flatten!(B, A)
            mapreduce(vec, vcat, A) == vec(B)
        end
    end
end
