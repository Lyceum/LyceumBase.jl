#include("src/SpecialArrays/SpecialArrays.jl")
using LyceumBase.SpecialArrays
using Random

x = rand(2,3,4)

s1 = SpecialArrays.Slices(x)
s2 = SpecialArrays.Slices(x, 1)
s3 = SpecialArrays.Slices(x, 1, 2)
s4 = SpecialArrays.Slices(x, 1, 2, 3)
nv = SpecialArrays.NestedView{2}(x)

s = SpecialArrays.Slices(rand(2,3,4), (1, 2))



e1 = rand!(SpecialArrays.ElasticArray{Float64}(undef, 2, 3))
e2 = rand!(SpecialArrays.ElasticArray{Float64}(undef, 2, 3))
se1 = SpecialArrays.Slices(e1, 1)
se2 = SpecialArrays.Slices(e2, 1)
x = [rand!(Array(el)) for el in se1]

@info "" size(x) size(se1)
@info "" SpecialArrays.innersize(x) SpecialArrays.innersize(se1)
copyto!(se1, x)


A = NestedView{1}(rand(2,3,40))
B = NestedView{1}(rand(2,3,40))
x = [rand!(Array(el)) for el in A]

#@btime copyto!($A, $x) evals=1 samples=5
#@btime SpecialArrays.mycopyto!($A, $x) evals=1 samples=5
#@btime copyto!($x, $A) evals=1 samples=5
#@btime SpecialArrays.mycopyto!($x, $A) evals=1 samples=5
const AbsArr{T,N} = AbstractArray{T,N}
using LyceumBase.LyceumCore
function g(A)
    SpecialArrays.nestedview(A, 2, true)
end
function h(A)
    SpecialArrays.nestedview(A, 2, True())
end