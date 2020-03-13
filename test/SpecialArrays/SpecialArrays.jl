module SpecialArraysTest
    using Test, Random

    import AxisArrays
    using UnsafeArrays

    import ..LyceumBase: SpecialArrays
    using .SpecialArrays
    using .SpecialArrays: ncolons, check_nestedarray_parameters, _maybe_unsqueeze, NestedVector

    @testset "NestedArray" begin
        include("nestedarrays/functions.jl")
        include("nestedarrays/nestedview.jl")
    end

    @testset "ElasticArray" begin
        include("elasticarray.jl")
    end

    @testset "slices.jl" begin
        include("slices.jl")
    end

end