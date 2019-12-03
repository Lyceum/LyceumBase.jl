abstract type AbstractEnv end

const EnvSpaces = NamedTuple{
    (:statespace, :observationspace, :actionspace, :rewardspace, :evaluationspace),
}



"""
    statespace(env::AbstractEnv) --> Shapes.AbstractShape

Returns a subtype of [`AbstractShape`](@ref Shapes.AbstractShape)
describing the state space of `env`.
"""
@mustimplement statespace(env::AbstractEnv)

@mustimplement getstate!(state, env::AbstractEnv)

getstate(env::AbstractEnv) = (s = allocate(statespace(env)); getstate!(s, env); s)



"""
    observationspace(env::AbstractEnv) --> Shapes.AbstractShape

Returns a subtype of [`AbstractShape`](@ref Shapes.AbstractShape)
describing the observationspace space of `env`.
"""
@mustimplement observationspace(env::AbstractEnv)

@mustimplement getobs!(observation, env::AbstractEnv)

getobs(env::AbstractEnv) = (o = allocate(observationspace(env)); getobs!(o, env); o)



"""
    actionspace(env::AbstractEnv) --> Shapes.AbstractShape

Returns a subtype of [`AbstractShape`](@ref Shapes.AbstractShape)
describing the action space of `env`.
"""
@mustimplement actionspace(env::AbstractEnv)

@mustimplement getaction!(action, env::AbstractEnv)

@mustimplement setaction!(env::AbstractEnv, action)

getaction(env::AbstractEnv) = (a = allocate(actionspace(env)); getaction!(a, env); a)



"""
    rewardspace(env::AbstractEnv) --> Shapes.AbstractShape

Returns a subtype of [`AbstractShape`](@ref Shapes.AbstractShape)
describing the reward space of `env`.
"""
rewardspace(env::AbstractEnv) = ScalarShape{Float64}()
@mustimplement getreward(env::AbstractEnv)



"""
    evaluationspace(env::AbstractEnv) --> Shapes.AbstractShape

Returns a subtype of [`AbstractShape`](@ref Shapes.AbstractShape)
describing the evaluation space of `env`.

The default evaluation space for `<: `[`AbstractEnv`](@ref LyceumBase.AbstractEnv)
is a [`ScalarShape{Float64}`](@ref Shapes.ScalarShape).
"""
evaluationspace(env::AbstractEnv) = ScalarShape{Float64}()
@mustimplement geteval(env::AbstractEnv)



"""
    reset!(env::AbstractEnv[, state[, action]])

Resets the environment.
    1. reset!(env) resets to the default state of `env`, zero actions.
    2. reset!(env, state) resets to `state`, zero actions.
    2. reset!(env, state, action) resets to `state` and `action`.
"""
@mustimplement reset!(env::AbstractEnv)
@mustimplement reset!(env::AbstractEnv, state)
reset!(env::AbstractEnv, state, action) = (reset!(env, state); setaction!(env, action); env)
@mustimplement randreset!(env::AbstractEnv)



@mustimplement step!(env::AbstractEnv)

function step!(env_t::AbstractEnv, action_t)
    setaction!(env_t, action_t)
    step!(env_t)
    # reward/eval are functions of: (env_t, c_t, env_tp1)
    (reward = getreward(env_t), eval = geteval(env_t), done = isdone(env_t))
end

isdone(env::AbstractEnv) = false



"""
    sharedmemory_envs(::Type{<:EnvType}, n, args...; kwargs...)

Returns `n` instances of EnvType, optionally sharing data.
Defaults to `Tuple(EnvType(args...; kwargs...) for _ in 1:n)`.

"""
function sharedmemory_envs(EnvType::Type{<:AbstractEnv}, n::Integer, args...; kwargs...)
    n > 0 || throw(ArgumentError("n must be > 0"))
    Tuple(EnvType(args...; kwargs...) for _ = 1:n)
end



"""
    Base.time(env)

Returns the current time using the `env` clock.
"""
@mustimplement Base.time(::AbstractEnv)

@mustimplement timestep(::AbstractEnv)
@mustimplement effective_timestep(::AbstractEnv)


function spaces(env::AbstractEnv)
    EnvSpaces((
        statespace(env),
        observationspace(env),
        actionspace(env),
        rewardspace(env),
        evaluationspace(env),
    ))
end
