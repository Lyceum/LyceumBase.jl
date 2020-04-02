module TestGeom

include("../preamble.jl")

@testset "SPoint3D" begin
    @test_throws ArgumentError SPoint3D(rand(4, 10), 3)
    @test_throws BoundsError SPoint3D(rand(3, 10), 11)
    A = rand(Float32, 3, 10)
    @test @inferred(SPoint3D(A, 5)) == A[:, 5]
    @test eltype(A) === Float32
    @test eltype(@inferred(SPoint3D{Float64}(A, 5))) === Float64
end

@testset "MPoint3D" begin
    @test_throws ArgumentError MPoint3D(rand(4, 10), 3)
    @test_throws BoundsError MPoint3D(rand(3, 10), 11)
    A = rand(Float32, 3, 10)
    @test @inferred(MPoint3D(A, 5)) == A[:, 5]
    @test eltype(A) === Float32
    @test eltype(@inferred(MPoint3D{Float64}(A, 5))) === Float64
end

end # module
