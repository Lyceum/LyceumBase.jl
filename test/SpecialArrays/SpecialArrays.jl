module SpecialArraysTest
    using Test, Random

    import AxisArrays
    using UnsafeArrays
    using BenchmarkTools

    using LyceumBase.LyceumCore
    import ..LyceumBase: SpecialArrays
    using .SpecialArrays
    using .SpecialArrays: ncolons, check_nestedarray_parameters, _maybe_unsqueeze

    #@testset "functions" begin
    #    include("nestedarrays/functions.jl")
    #end

    #@testset "NestedArray" begin
    #    include("nestedarrays/nestedview.jl")
    #end

    #@testset "ElasticArray" begin
    #    include("elasticarray.jl")
    #end

    @testset "slices.jl" begin
        include("slices.jl")
    end

end