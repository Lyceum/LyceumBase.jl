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
include("./test/toyenv.jl")

ntimesteps = 50 * Threads.nthreads()
#env_kwargs = (max_length = 50, reward_scale = 5, step_hook = _ -> isodd(Threads.threadid()) && busyloop(0.001))
env_kwargs = (max_length = 50, reward_scale = 5, step_hook = _ -> busyloop(3e-5))
#env_kwargs = (max_length = 50, reward_scale = 5, step_hook = _ -> busyloop(3e-4))
sampler = EnvironmentSampler(n -> ntuple(i -> ToyEnv(; env_kwargs...), n))
B1 = sample((a,o) -> a, sampler, ntimesteps, reset! = reset!, nthreads = 1)
B2 = sample((a,o) -> a, sampler, ntimesteps, reset! = reset!)
if true
b1 = @benchmark sample((a,o) -> a, $sampler, $ntimesteps, reset! = reset!, nthreads=1) evals=1 samples=80
display(b1)
b2 = @benchmark sample((a,o) -> a, $sampler, $ntimesteps, reset! = reset!) evals=1 samples=300
display(b2)
@info mean(b1.times) / mean(b2.times)
end
end
