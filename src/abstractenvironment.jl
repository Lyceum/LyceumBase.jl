"""
    AbstractEnvironment

Supertype for all environments.
"""
abstract type AbstractEnvironment end

const EnvSpaces = NamedTuple{
    (:statespace, :obsspace, :actionspace, :rewardspace, :evalspace),
}


"""
    statespace(env::AbstractEnvironment) --> Shapes.AbstractShape

Returns a subtype of `Shapes.AbstractShape` describing the state space of `env`.

See also: [`getstate!`](@ref), [`setstate!`](@ref), [`getstate`](@ref).
"""
@mustimplement statespace(env::AbstractEnvironment)

"""
    getstate!(state, env::AbstractEnvironment)

Store the current state of `env` in `state`, where `state` conforms to the state space
returned by `statespace(env)`.

See also: [`statespace`](@ref), [`setstate!`](@ref), [`getstate`](@ref).
"""
@mustimplement getstate!(state, env::AbstractEnvironment)

"""
    getstate(env::AbstractEnvironment)

Get the current state of `env`. The returned value will be an object
conforming to the state space returned by `statespace(env)`.

See also: [`statespace`](@ref), [`getstate!`](@ref), [`setstate!`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    [`statespace`](@ref) and [`getstate!`](@ref), which are used internally by `getstate`.
"""
@propagate_inbounds function getstate(env::AbstractEnvironment)
    s = allocate(statespace(env))
    getstate!(s, env)
    s
end

"""
    setstate!(env::AbstractEnvironment, state)

Set the state of `env` to `state`, where `state` conforms to the state space returned
by `statespace(env)`.

See also: [`statespace`](@ref), [`getstate!`](@ref), [`getstate`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes must guarantee that calls to
    other "getter" functions (e.g. `getreward`) after a call to `setstate!` reflect the
    new, passed-in state.
"""
@mustimplement setstate!(env::AbstractEnvironment, state)


"""
    obsspace(env::AbstractEnvironment) --> Shapes.AbstractShape

Returns a subtype of `Shapes.AbstractShape` describing the observation space of `env`.

See also: [`getobs!`](@ref), [`getobs`](@ref).
"""
@mustimplement obsspace(env::AbstractEnvironment)

"""
    getobs!(obs, env::AbstractEnvironment)

Store the current observation of `env` in `obs`, where `obs` conforms to the
observation space returned by `obsspace(env)`.

See also: [`obsspace`](@ref), [`getobs`](@ref).
"""
@mustimplement getobs!(obs, env::AbstractEnvironment)

"""
    getobs(env::AbstractEnvironment)

Get the current observation of `env`. The returned value will be an object
conforming to the observation space returned by `obsspace(env)`.

See also: [`obsspace`](@ref), [`getobs!`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    [`obsspace`](@ref) and [`getobs!`](@ref), which are used internally by `getobs`.
"""
@propagate_inbounds function getobs(env::AbstractEnvironment)
    o = allocate(obsspace(env))
    getobs!(o, env)
    o
end


"""
    actionspace(env::AbstractEnvironment) --> Shapes.AbstractShape

Returns a subtype of `Shapes.AbstractShape` describing the action space of `env`.

See also: [`getaction!`](@ref), [`setaction!`](@ref), [`getaction`](@ref).
"""
@mustimplement actionspace(env::AbstractEnvironment)

"""
    getaction!(action, env::AbstractEnvironment)

Store the current action of `env` in `action`, where `action` conforms to the action space
returned by `actionspace(env)`.

See also: [`actionspace`](@ref), [`setaction!`](@ref), [`getaction`](@ref).
"""
@mustimplement getaction!(action, env::AbstractEnvironment)

"""
    getaction(env::AbstractEnvironment)

Get the current action of `env`. The returned value will be an object
conforming to the action space returned by `actionspace(env)`.

See also: [`actionspace`](@ref), [`getaction!`](@ref), [`setaction!`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    [`actionspace`](@ref) and [`getaction!`](@ref), which are used internally by `getaction`.
"""
@propagate_inbounds function getaction(env::AbstractEnvironment)
    a = allocate(actionspace(env))
    getaction!(a, env)
    a
end

"""
    setaction!(env::AbstractEnvironment, action)

Set the action of `env` to `action`, where `action` conforms to the action space returned
by `actionspace(env)`.

See also: [`actionspace`](@ref), [`getaction!`](@ref), [`getaction`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes must guarantee that calls to
    other "getter" functions (e.g. `getreward`) after a call to `setaction!` reflect the
    new, passed-in action.
"""
@mustimplement setaction!(env::AbstractEnvironment, action)


"""
    rewardspace(env::AbstractEnvironment) --> Shapes.AbstractShape

Returns a subtype of `Shapes.AbstractShape` describing the reward space of `env`.
Defaults to `Shapes.ScalarShape{Float64}()`.

See also: [`getreward`](@ref).

!!! note
    Currently, only scalar spaces are supported (e.g. `Shapes.ScalarShape`).
"""
@inline rewardspace(env::AbstractEnvironment) = ScalarShape{Float64}()

"""
    getreward(state, action, observation, env::AbstractEnvironment)

Get the current reward of `env` as a function of `state`, `action`, and `observation`.
The returned value will be an object conforming to the reward space returned by `rewardspace(env)`.

See also: [`rewardspace`](@ref).

!!! note
    Currently, only scalar rewards are supported, so there is no in-place `getreward!`.

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should be careful to
    ensure that the result of `getreward` is purely a function of `state`/`action`/`observation`
    and not any internal, dynamic state contained in `env`.
"""
@mustimplement getreward(state, action, observation, env::AbstractEnvironment)

"""
    getreward(env::AbstractEnvironment)

Get the current reward of `env`.

Internally calls `getreward(getstate(env), getaction(env), getobs(env), env)`.

See also: [`rewardspace`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    `getreward(state, action, observation, env)`.
"""
@propagate_inbounds function getreward(env::AbstractEnvironment)
    getreward(getstate(env), getaction(env), getobs(env), env)
end


"""
    evalspace(env::AbstractEnvironment) --> Shapes.AbstractShape

Returns a subtype of `Shapes.AbstractShape` describing the evaluation space of `env`.
Defaults to `Shapes.ScalarShape{Float64}()`.

See also: [`geteval`](@ref).

!!! note
    Currently, only scalar evaluation spaces are supported (e.g. `Shapes.ScalarShape`).
"""
@inline evalspace(env::AbstractEnvironment) = ScalarShape{Float64}()

"""
    geteval(state, action, observation, env::AbstractEnvironment)

Get the current evaluation metric of `env` as a function of `state`, `action`,
and `observation`. The returned value will be an object conforming to the
evaluation space returned by `evalspace(env)`.

Often times reward functions are heavily "shaped" and hard to interpret.
For example, the reward function for bipedal walking may include root pose, ZMP terms, control costs, etc.,
while success can instead be simply evaluated by distance of the root along an axis. The evaluation
metric serves to fill this gap.

The default behavior is to return `getreward(state, action, observation, env::AbstractEnvironment)`.

See also: [`evalspace`](@ref).

!!! note
    Currently, only scalar evaluation metrics are supported, so there is no in-place `geteval!`.

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should be careful to
    ensure that the result of `geteval` is purely a function of `state`/`action`/`observation`
    and not any internal, dynamic state contained in `env`.
"""
@mustimplement geteval(state, action, observation, env::AbstractEnvironment)

"""
    geteval(env::AbstractEnvironment)

Get the current evaluation metric of `env`.

Internally calls `geteval(getstate(env), getaction(env), getobs(env), env)`.

See also: [`evalspace`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    `geteval(state, action, obs, env)`.
"""
@propagate_inbounds function geteval(env::AbstractEnvironment)
    geteval(getstate(env), getaction(env), getobs(env), env)
end


"""
    reset!(env::AbstractEnvironment)

Reset `env` to a fixed, initial state with zero/passive controls.
"""
@mustimplement reset!(env::AbstractEnvironment)

"""
    randreset!([rng::Random.AbstractRNG, ], env::AbstractEnvironment)

Reset `env` to a random state with zero/passive controls.

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    randreset!(rng, env).
"""
@mustimplement randreset!(rng::Random.AbstractRNG, env::AbstractEnvironment)
@propagate_inbounds randreset!(env::AbstractEnvironment) = randreset!(Random.default_rng(), env)


"""
    step!(env::AbstractEnvironment)

Advance `env` forward by one timestep.

See also: [`timestep`](@ref).
"""
@mustimplement step!(env::AbstractEnvironment)


"""
    isdone(state, action, observation, env::AbstractEnvironment) --> Bool

Returns `true` if `state`, `action`, and `observation` meet an
early termination condition for `env`. Defaults to `false`.

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should be careful to
    ensure that the result of `isdone` is purely a function of `state`/`action`/`observation`
    and not any internal, dynamic state contained in `env`.
"""
isdone(state, action, obs, env::AbstractEnvironment) = false

"""
    isdone(env::AbstractEnvironment)

Returns `true` if `env` has met an early termination condition.

Internally calls `isdone(getstate(env), getaction(env), getobs(env), env)`.

!!! note
    Implementers of custom `AbstractEnvironment` subtypes should implement
    `isdone(state, action, obs, env)`.
"""
@propagate_inbounds function isdone(env::AbstractEnvironment)
    isdone(getstate(env), getaction(env), getobs(env), env)
end


"""
    Base.time(env::AbstractEnvironment)

Returns the current simulation time, in seconds, of `env`. By convention,
`time(env)` should return zero after a call to `reset!(env)` or `randreset!(env)`.

See also: [`timestep`](@ref).
"""
@mustimplement Base.time(env::AbstractEnvironment)

"""
    timestep(env::AbstractEnvironment)

Return the internal simulation timestep, in seconds, of `env`.

See also: [`Base.time`](@ref).

# Examples

```julia
env = FooEnv()
reset!(env)
t1 = time(env)
step!(env)
t2 = time(env)
@assert timestep(env) == (t2 - t1)
```
"""
@mustimplement timestep(env::AbstractEnvironment)

"""
    spaces(env::AbstractEnvironment)

Return a `NamedTuple` of containing all of `env`'s spaces.

# Examples

```julia
env = FooEnv()
sp = spaces(env)
@assert statespace(env) == sp.statespace
```
"""
function spaces(env::AbstractEnvironment)
    EnvSpaces((
        statespace(env),
        obsspace(env),
        actionspace(env),
        rewardspace(env),
        evalspace(env),
    ))
end