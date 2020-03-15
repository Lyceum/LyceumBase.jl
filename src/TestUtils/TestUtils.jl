module TestUtils

using Test: Test, @testset, @test, @inferred

using BenchmarkTools: BenchmarkTools

using ..LyceumBase


export @test_inferred, @test_noalloc


macro test_inferred(ex)
    ex = quote
        $Test.@test (($Test.@inferred $ex); true)
    end
    esc(ex)
end

macro test_noalloc(ex)
    ex = quote
        local nbytes = $BenchmarkTools.@ballocated $ex samples = 1 evals = 1
        $Test.@test iszero(nbytes)
    end
    esc(ex)
end

export testenv_correctness, testenv_allocations, testenv_inferred
include("abstractenvironment.jl")

end
