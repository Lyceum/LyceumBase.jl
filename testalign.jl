module Mod
include("src/LyceumBase.jl")
using .LyceumBase
using .LyceumBase.SpecialArrays
using .LyceumBase.LyceumCore

using Random
using UnsafeArrays
using StaticNumbers
using AxisArrays: AxisArray
using Test

x = [rand(2,3) for _=1:4]
A = FlattenedArray(x)

end
