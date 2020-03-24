module TestUtil

using Test: Test, @testset, @test, @inferred

using BenchmarkTools: BenchmarkTools, memory
using MacroTools: MacroTools

using ..LyceumBase


export @qe
export @test_inferred, @test_noalloc


"""
    @qe [expression]

Equivalent to:

    quote
        \$(MacroTools.striplines(esc(expression)))
    end
end
"""
macro qe(ex)
  Expr(:quote, MacroTools.striplines(esc(ex)))
end

macro test_inferred(ex)
    @qe begin
        $Test.@test (($Test.@inferred $ex); true)
    end
end

macro test_noalloc(ex)
    @qe begin
        local nbytes = $memory($BenchmarkTools.@benchmark $ex samples = 1 evals = 1)
        $iszero(nbytes) ? true : $error("Allocated $(BenchmarkTools.prettymemory(nbytes))")
    end
end


export testenv_correctness, testenv_allocations, testenv_inferred
include("abstractenvironment.jl")

end
