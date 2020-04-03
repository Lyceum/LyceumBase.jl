Base.@kwdef mutable struct ToyEnv <: AbstractEnvironment
    s::Int = 0
    a::Int = 0
    t::Int = 0
    max_length::Int = typemax(Int)
    step_hook = identity
    reward_scale::Float64 = 1
end


LyceumBase.statespace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getstate!(s, e::ToyEnv) = s .= e.s
LyceumBase.setstate!(e::ToyEnv, s) = (e.s = s[]; e)

LyceumBase.obsspace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getobs!(o, e::ToyEnv) = o .= e.t

LyceumBase.actionspace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getaction!(a, e::ToyEnv) = a .= e.a
LyceumBase.setaction!(e::ToyEnv, a) = (e.a = a[]; e)

LyceumBase.getreward(s, a, o, e::ToyEnv) = s[] * e.reward_scale


function LyceumBase.reset!(e::ToyEnv)
    e.s = e.a = e.t = 0
    return e
end

function LyceumBase.randreset!(rng::AbstractRNG, e::ToyEnv)
    e.s = rand(rng, -5:5)
    e.t = 0
    return e
end


function LyceumBase.step!(e::ToyEnv)
    e.s += e.a
    e.t += 1
    e.step_hook(e)
    return e
end

LyceumBase.isdone(s, o, e::ToyEnv) = o[] >= (e.max_length - 1)

Base.time(e::ToyEnv) = e.t
LyceumBase.timestep(e::ToyEnv) = e.dt

function busyloop(dt::Real)
    t0 = time()
    while time() - t0 < dt
    end
end
