@testset "functions" begin
    @testset "inner" begin
        let A = [zeros(2,4) for _=1:6]
            @test @inferred(innereltype(A)) == Float64

            @test @inferred(innerndims(A)) == 2

            @test @inferred(inneraxes(A)) == (Base.OneTo(2), Base.OneTo(4))
            @test @inferred(inneraxes(A, 1)) == Base.OneTo(2)
            @test @inferred(inneraxes(A, 2)) == Base.OneTo(4)
            @test @inferred(inneraxes(A, 3)) == Base.OneTo(1)

            @test @inferred(innersize(A)) == (2, 4)
            @test @inferred(innersize(A, 1)) == 2
            @test @inferred(innersize(A, 2)) == 4
            @test @inferred(innersize(A, 3)) == 1

            @test @inferred(innerlength(A)) == 8
        end

        let A = Vector{Vector{Int}}()
            @test @inferred(innereltype(A)) == Int
            @test @inferred(innereltype(typeof(A))) == Int

            @test @inferred(innerndims(A)) == 1
            @test @inferred(innerndims(typeof(A))) == 1

            @test @inferred(inneraxes(A)) == (Base.OneTo(0), )
            @test @inferred(inneraxes(A, 1)) == Base.OneTo(0)
            @test @inferred(inneraxes(A, 2)) == Base.OneTo(1)

            @test @inferred(innersize(A)) == (0, )
            @test @inferred(innersize(A, 1)) == 0
            @test @inferred(innersize(A, 2)) == 1

            @test @inferred(innerlength(A)) == 0
        end

        let A = [zeros(2), zeros(4, 6)]
            # ndims(eltype(A)) throws a MethodError on A
            @test_throws MethodError @inferred(innerndims(A))
        end

        let A = [zeros(2), zeros(3)]
            @test @inferred(innereltype(A)) == Float64
            @test_throws DimensionMismatch @inferred(inneraxes(A))
            @test_throws DimensionMismatch @inferred(innersize(A))
            @test_throws DimensionMismatch @inferred(innerlength(A))
        end

        let A = [zeros(Int, 2), zeros(Float64, 2)]
            # A is promoted to Vector{Vector{Float64}}
            @test innereltype(A) == Float64
            # should still work even if inner eltype is inconsistent
            @test innerndims(A) == 1
            @test inneraxes(A) == (Base.OneTo(2), )
            @test innersize(A) == (2, )
        end

        let A = [zeros(Int, 2), zeros(Float64, 4, 6)]
            # should still work even if inner dimensionality is inconsistent
            @test innereltype(A) == Any
        end
    end

    @testset "flatten/flatten!" begin
        @test_throws DimensionMismatch flatten!(rand(2,3,4), [rand(2)])
        @test_throws DimensionMismatch flatten!(rand(2,3,4), [rand(2,3,4)])
        @test_throws DimensionMismatch flatten!(rand(2,3,4), [rand(2,2)])
        @test_throws ArgumentError flatten!(rand(2,3,4), [rand(2,3) for _=1:5])

        @testset "M=$M N=$N" for M=0:3, N=0:3
            nested, flat = randNA(M, N)
            rand!(flat)
            @test flatten!(flat, nested) === flat
            @test all(CartesianIndices(axes(nested))) do I
                _maybe_unsqueeze(nested[I]) == flat[ncolons(M)..., Tuple(I)...]
            end
        end
    end

    @testset "nest/nest!" begin
        @test_throws DimensionMismatch nest!([rand(2)], rand(2,3,4))
        @test_throws DimensionMismatch nest!([rand(2,3,4)], rand(2,3,4))
        @test_throws DimensionMismatch nest!([rand(2,2)], rand(2,3,4))
        @test_throws ArgumentError nest!([rand(2,3)], rand(2,3,2))

        @testset "M=$M N=$N" for M=0:3, N=0:3
            nested, flat = randNA(M, N)
            rand!(flat)
            @test nest!(nested, flat) === nested
            @test all(CartesianIndices(axes(nested))) do I
                _maybe_unsqueeze(nested[I]) == flat[ncolons(M)..., Tuple(I)...]
            end
        end
    end

end
