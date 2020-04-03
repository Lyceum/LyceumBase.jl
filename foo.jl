using LyceumBase
using Random
using Shapes
include("test/toyenv.jl")

e=ToyEnv()
sampler = EnvironmentSampler(n -> [ToyEnv(step_hook = _ -> busyloop(0.001)) for _=1:n])

@btime begin
sample($sampler, 100, reset! = reset!, Hmax = 10, nthreads=1) do a, s, o
    a .= Threads.threadid()
end
end
@btime begin
sample($sampler, 100, reset! = reset!, Hmax = 10) do a, s, o
    a .= Threads.threadid()
end
end

#V1 = sample(sampler, 100, reset! = reset!, Hmax = 10, nthreads=1) do a, s, o
#    a .= Threads.threadid()
#end
#V2 = sample(sampler, 30, reset! = reset!, Hmax = 5) do a, s, o
#    a .= Threads.threadid()
#end


nothing
