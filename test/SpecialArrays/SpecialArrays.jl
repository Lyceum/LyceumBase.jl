module SpecialArraysTest
    using Test, Random

    import ..LyceumBase: SpecialArrays
    using .SpecialArrays
    using .SpecialArrays: ncolons, check_nestedarray_parameters, _maybe_unsqueeze, NestedVector

    using UnsafeArrays

    @testset "NestedArray" begin
        include("nestedarrays/functions.jl")
        include("nestedarrays/nestedview.jl")
    end

    @testset "ElasticBuffer" begin
        include("elasticbuffer.jl")
    end

end