@testset "functions" begin
    @testset "inner_size" begin
        @test @inferred(inner_size([[1, 2, 3], [4, 5, 6]])) == (3,)
        @test @inferred(inner_length([[1, 2, 3], [4, 5, 6]])) == 3
        @test @inferred(inner_eltype([[1, 2, 3], [4, 5, 6]])) == Int

        @test @inferred(inner_size([[]])) == (0,)
        @test @inferred(inner_length([[]])) == 0
        @test_throws DimensionMismatch @inferred(inner_size([[1, 2, 3], [4, 5]]))
    end
end
