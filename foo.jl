module M

include("src/LyceumBase.jl")
using .LyceumBase

using Distributions: Uniform, sample
using LinearAlgebra
using LyceumDevTools.TestUtil
using Random
using Shapes
using Test
using BenchmarkTools
using Profile
include("./test/toyenv.jl")

using StructArrays
using SpecialArrays

env_kwargs = (max_length = 10, reward_scale = 5, step_hook = _ -> busyloop(3e-5))
ntimesteps = env_kwargs.max_length * Threads.nthreads() * 10
sampler = EnvironmentSampler(n -> ntuple(i -> ToyEnv(; env_kwargs...), n))
B = TrajectoryBuffer(ToyEnv(;env_kwargs...), sizehint = 1024)
println("------------------------")
@info "YOOO"
sample!(B, (a,o) -> a, sampler, ntimesteps, reset! = reset!, Hmax=env_kwargs.max_length)
@info map(LyceumBase.nsamples, sampler.buffers)


B2 = TrajectoryBuffer(ToyEnv(;env_kwargs...), sizehint = ntimesteps)
LyceumBase.collate!(B2, sampler.buffers, ntimesteps)
@btime begin
    LyceumBase.collate!(_B, $sampler.buffers, $ntimesteps)
end evals=1 samples=100 setup=(_B=TrajectoryBuffer(ToyEnv(;$env_kwargs...), sizehint = ntimesteps))

if true
b1 = @benchmark begin
    sample!(_B, (a,o) -> a, $sampler, $ntimesteps, reset! = reset!, Hmax=$env_kwargs.max_length, nthreads=1)
end evals=1 samples=80 setup=(_B=TrajectoryBuffer(ToyEnv(;$env_kwargs...), sizehint = ntimesteps))
#end setup=(_B=TrajectoryBuffer(ToyEnv(;$env_kwargs...), sizehint = 1))
display(b1)

b2 = @benchmark begin
    sample!(_B, (a,o) -> a, $sampler, $ntimesteps, reset! = reset!, Hmax=$env_kwargs.max_length)
end evals=1 samples=300 setup=(_B=TrajectoryBuffer(ToyEnv(;$env_kwargs...), sizehint = ntimesteps))
#end setup=(_B=TrajectoryBuffer(ToyEnv(;$env_kwargs...), sizehint = 1))
display(b2)

@info mean(b1.times) / mean(b2.times)
end

end # end module
