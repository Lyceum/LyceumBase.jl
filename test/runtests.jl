module TestLyceumBase

#include("preamble.jl")
using Test

include("testutil.jl")

@includetests ProgressTestSet "LyceumBase"

end # module
