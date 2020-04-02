const REWARD_SCALE = 2
const MAX_LENGTH = 50


mutable struct ToyEnv <: AbstractEnvironment
    s::Int
    a::Int
    t::Int
end

ToyEnv() = ToyEnv(0, 0, 0)


LyceumBase.statespace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getstate!(s, e::ToyEnv) = s .= e.s
LyceumBase.setstate!(e::ToyEnv, s) = (e.s = s[]; e)

LyceumBase.obsspace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getobs!(o, e::ToyEnv) = o .= e.t

LyceumBase.actionspace(::ToyEnv) = VectorShape(Int, 1)
LyceumBase.getaction!(a, e::ToyEnv) = a .= e.a
LyceumBase.setaction!(e::ToyEnv, a) = (e.a = a[]; e)

LyceumBase.getreward(s, a, o, e::ToyEnv) = s[] * REWARD_SCALE


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
    isodd(e.a) && busyloop(0.05)
    return e
end

LyceumBase.isdone(s, o, e::ToyEnv) = o[] > 10

Base.time(e::ToyEnv) = e.t
LyceumBase.timestep(e::ToyEnv) = e.dt

function busyloop(dt::Real)
    t0 = time()
    while time() - t0 < dt
    end
end
