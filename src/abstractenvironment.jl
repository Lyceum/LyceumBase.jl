"""
    $(TYPEDEF)

Supertype for all Lyceum environments.
"""
abstract type AbstractEnvironment end


"""
    $(TYPEDSIGNATURES)

Return a subtype of `Shapes.AbstractShape` describing the state space of `env`.

See also: [`getstate!`](@ref), [`getstate`](@ref), [`setstate!`](@ref).
"""
@mustimplement statespace(env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Store the current state of `env` in `s`.

See also: [`statespace`](@ref), [`getstate`](@ref), [`setstate!`](@ref).
"""
@mustimplement getstate!(s, env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Get the current state of `env`.

See also: [`statespace`](@ref), [`getstate!`](@ref), [`setstate!`](@ref).

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement [`statespace`](@ref) and
    [`getstate!`](@ref), which are used internally by `getstate`.
"""
@propagate_inbounds function getstate(env::AbstractEnvironment)
    s = allocate(statespace(env))
    getstate!(s, env)
    return s
end

"""
    $(TYPEDSIGNATURES)

Set the current state of `env` to `s`.

See also: [`statespace`](@ref), [`getstate!`](@ref), [`getstate`](@ref).

!!! note
    Implementers of `AbstractEnvironment` subtypes must guarantee that calls to
    other "getter" functions (e.g. `getreward`) after a call to `setstate!` reflect the
    new state `s`.
"""
@mustimplement setstate!(env::AbstractEnvironment, s)


"""
    $(TYPEDSIGNATURES)

Return a subtype of `Shapes.AbstractShape` describing the observation space of `env`.

See also: [`getobservation!`](@ref), [`getobservation`](@ref).
"""
@mustimplement observationspace(env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Store the current observation of `env` in `o`.

See also: [`observationspace`](@ref), [`getobservation`](@ref).
"""
@mustimplement getobservation!(o, env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Get the current observation of `env`.

See also: [`observationspace`](@ref), [`getobservation!`](@ref).

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement [`observationspace`](@ref) and
    [`getobservation!`](@ref), which are used internally by `getobservation`.
"""
@propagate_inbounds function getobservation(env::AbstractEnvironment)
    o = allocate(observationspace(env))
    getobservation!(o, env)
    o
end


"""
    $(TYPEDSIGNATURES)

Returns a subtype of `Shapes.AbstractShape` describing the action space of `env`.

See also: [`getaction!`](@ref), [`getaction`](@ref), [`setaction!`](@ref).
"""
@mustimplement actionspace(env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Store the current action of `env` in `a`.

See also: [`actionspace`](@ref), [`getaction`](@ref), [`setaction!`](@ref).
"""
@mustimplement getaction!(a, env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Get the current action of `env`.

See also: [`actionspace`](@ref), [`getaction!`](@ref), [`setaction!`](@ref).

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement [`actionspace`](@ref) and
    [`getaction!`](@ref), which are used internally by `getaction`.
"""
@propagate_inbounds function getaction(env::AbstractEnvironment)
    a = allocate(actionspace(env))
    getaction!(a, env)
    a
end

"""
    $(TYPEDSIGNATURES)

Set the current action of `env` to `a`.

See also: [`actionspace`](@ref), [`getaction!`](@ref), [`getaction`](@ref).

!!! note
    Implementers of custom `AbstractEnvironment` subtypes must guarantee that calls to
    other "getter" functions (e.g. `getreward`) after a call to `setaction!` reflect the
    new action `a`.
"""
@mustimplement setaction!(env::AbstractEnvironment, a)


"""
    $(TYPEDSIGNATURES)

Returns a subtype of `Shapes.AbstractShape` describing the reward space of `env`.
Defaults to `Shapes.ScalarShape(Float64)`.

See also: [`getreward`](@ref).

!!! note
    Currently, only scalar spaces are supported.
"""
@inline rewardspace(env::AbstractEnvironment) = ScalarShape(Float64)

"""
    $(TYPEDSIGNATURES)

Get the current reward of `env` as a function of state `s`, action `a`, and observation `o`.

See also: [`rewardspace`](@ref).

!!! note
    Currently, only scalar rewards are supported, so there is no in-place `getreward!`.

!!! note
    Implementers of `AbstractEnvironment` subtypes should be careful to ensure that the result
    of `getreward` is purely a function of `s`/`a`/`o` and and not any internal, dynamic state
    contained in `env`.
"""
@mustimplement getreward(s, a, o, env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Get the current reward of `env`.

See also: [`rewardspace`](@ref).

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement `getreward(s, a, o, env)`.
"""
@propagate_inbounds function getreward(env::AbstractEnvironment)
    getreward(getstate(env), getaction(env), getobservation(env), env)
end


"""
    $(TYPEDSIGNATURES)

Reset `env` to a fixed, initial state with passive dynamics (i.e. a "zero" action).
"""
@mustimplement reset!(env::AbstractEnvironment)

"""
    randreset!([rng::Random.AbstractRNG, ], env::AbstractEnvironment)

Reset `env` to a random state with passive dynamics (i.e. a "zero" action).

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement randreset!(rng, env).
"""
@mustimplement randreset!(rng::Random.AbstractRNG, env::AbstractEnvironment)
@propagate_inbounds randreset!(env::AbstractEnvironment) = randreset!(Random.default_rng(), env)


"""
    $(TYPEDSIGNATURES)

Advance `env` forward by one timestep.

See also: [`timestep`](@ref).
"""
@mustimplement step!(env::AbstractEnvironment)


"""
    $(TYPEDSIGNATURES)

Returns `true` if state `s` and observation `o` meet an early termination condition for `env`.
Defaults to `false`.

!!! note
    Implementers of `AbstractEnvironment` subtypes should be careful to ensure that the result of
    `isdone` is purely a function of `s` and `o` and not any internal, dynamic state contained
    in `env`.
"""
isdone(s, o, env::AbstractEnvironment) = false

"""
    $(TYPEDSIGNATURES)

Returns `true` if `env` has met an early termination condition.

Internally calls `isdone(getstate(env), getobservation(env), env)`.

!!! note
    Implementers of `AbstractEnvironment` subtypes should implement `isdone(s, o, env)`.
"""
@propagate_inbounds function isdone(env::AbstractEnvironment)
    isdone(getstate(env), getobservation(env), env)
end


"""
    $(TYPEDSIGNATURES)

Returns the current simulation time, in seconds, of `env`.

See also: [`timestep`](@ref).
"""
@mustimplement Base.time(env::AbstractEnvironment)

"""
    $(TYPEDSIGNATURES)

Return the internal simulation timestep, in seconds, of `env`.

See also: [`Base.time`](@ref).
"""
@mustimplement timestep(env::AbstractEnvironment)


"""
    $(TYPEDSIGNATURES)

Return a `NamedTuple` of containing all of `env`'s spaces.

# Examples
```julia
env = FooEnv()
sp = spaces(env)
@assert statespace(env) == sp.statespace
```
"""
function spaces(env::AbstractEnvironment)
    (
        statespace = statespace(env),
        observationspace = observationspace(env),
        actionspace = actionspace(env),
        rewardspace = rewardspace(env),
    )
end
