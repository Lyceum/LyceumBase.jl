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

Returns a subtype of [`Shapes.AbstractShape`](@ref) describing the state space of `env`.

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

Returns a subtype of [`Shapes.AbstractShape`](@ref) describing the observation space of `env`.

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

Returns a subtype of [`Shapes.AbstractShape`](@ref) describing the action space of `env`.

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

Returns a subtype of [`Shapes.AbstractShape`](@ref) describing the reward space of `env`.
Defaults to `Shapes.ScalarShape{Float64}()`.

See also: [`getreward`](@ref).

!!! note
    Currently, only scalar spaces are supported (e.g. [`Shapes.ScalarShape`](@ref)).
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

Returns a subtype of [`Shapes.AbstractShape`](@ref) describing the evaluation space of `env`.
Defaults to `Shapes.ScalarShape{Float64}()`.

See also: [`geteval`](@ref).

!!! note
    Currently, only scalar evaluation spaces are supported (e.g. [`Shapes.ScalarShape`](@ref)).
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

# TODO factor out this whole section to something like LyceumTestUtils (along with Tools.@noalloc + deps)

function _trajectory(e::AbstractEnvironment, T::Integer)
    (
        states = Array(undef, statespace(e), T),
        obses = Array(undef, obsspace(e), T),
        acts = Array(undef, actionspace(e), T),
        rews = Array(undef, rewardspace(e), T),
        evals = Array(undef, evalspace(e), T)
    )
end

function _rollout(e::AbstractEnvironment, actions::AbstractMatrix)
    T = size(actions, 2)
    traj = _trajectory(e, T)
    traj.acts .= actions
    for t=1:T
        st = view(traj.states, :, t)
        at = view(traj.acts, :, t)
        ot = view(traj.obses, :, t)

        getstate!(st, e)
        getobs!(ot, e)
        setaction!(e, at)
        traj.rews[t] = getreward(st, at, ot, e)
        traj.evals[t] = geteval(st, at, ot, e)
        step!(e)
    end
    traj
end

macro noalloc(expr)
    quote
        local tmp = @benchmark $expr samples = 1 evals = 1
        iszero(tmp.allocs)
    end
end

function test_env(etype::Type{<:AbstractEnvironment}, args...; kwargs...)
    @testset "Testing $etype\n    Args: $args.\n    Kwargs: $kwargs" begin
        makeenv() = etype(args...; kwargs...)::AbstractEnvironment
        @testset "Interface" begin

            @test isconcretetype(typeof(makeenv()))

            @testset "time consistency" begin
                e = makeenv()
                @test time(e) isa Float64
                t1 = time(e)
                @test time(e) === t1
                @test step!(e) === e
                t2 = time(e)
                @test (t2 - t1) == timestep(e)
            end

            # For each of state/action/observation/reward/eval, test:
            # 1. return type of getfoo(env) matches foospace(env)
            # 2. getfoo!(x, env) returns x
            # 3. getfoo!(x, env) == getfoo(env)
            # 4. if settable, that setfoo!(env, x) returns env
            # 5. if settable, that getfoo(env) == x after setfoo!(env, x)
            # 6. all functions are inferrable

            @testset "state" begin
                let e = makeenv()
                    @test statespace(e) isa Shapes.AbstractVectorShape
                    @test eltype(getstate(e)) == eltype(statespace(e))
                    @test axes(getstate(e)) == axes(statespace(e))
                end

                @test let e = makeenv(), x = rand(statespace(e))
                    x === getstate!(x, e)
                end
                @test let e = makeenv()
                    getstate!(rand(statespace(e)), e) == getstate(e)
                end

                @test let e = makeenv()
                    e === setstate!(e, rand(statespace(e)))
                end
                @test let e = makeenv(), x = rand(statespace(e))
                    setstate!(e, x)
                    x == getstate!(rand(statespace(e)), e) == getstate(e)
                end
            end

            @testset "action" begin
                let e = makeenv()
                    @test actionspace(e) isa Shapes.AbstractVectorShape
                    @test eltype(getaction(e)) == eltype(actionspace(e))
                    @test axes(getaction(e)) == axes(actionspace(e))
                end

                @test let e = makeenv(), x = rand(actionspace(e))
                    x === getaction!(x, e)
                end
                @test let e = makeenv()
                    getaction!(rand(actionspace(e)), e) == getaction(e)
                end

                @test let e = makeenv()
                    e === setaction!(e, rand(actionspace(e)))
                end
                @test let e = makeenv(), x = rand(actionspace(e))
                    setaction!(e, x)
                    x == getaction!(rand(actionspace(e)), e) == getaction(e)
                end
            end

            @testset "observation" begin
                let e = makeenv()
                    @test obsspace(e) isa Shapes.AbstractVectorShape
                    @test eltype(getobs(e)) == eltype(obsspace(e))
                    @test axes(getobs(e)) == axes(obsspace(e))
                end

                @test let e = makeenv(), x = rand(obsspace(e))
                    x === getobs!(x, e)
                end
                @test let e = makeenv()
                    getobs!(rand(obsspace(e)), e) == getobs(e)
                end
            end

            # For getreward, geteval, and isdone, additionally test that they are functions of the
            # passed in (state, action, observation) and not any internal data in the env.

            @testset "reward" begin
                let e = makeenv()
                    @test rewardspace(e) isa ScalarShape
                    @test typeof(getreward(e)) == eltype(rewardspace(e))
                end

                @test let e = makeenv(), s = getstate(e), a = getaction(e), o = getobs(e)
                    r1 = getreward(s, a, o, e)
                    r2 = getreward(e)
                    randreset!(e)
                    r3 = getreward(s, a, o, e)
                    r1 == r2 == r3
                end
            end

            @testset "eval" begin
                let e = makeenv()
                    @test evalspace(e) isa ScalarShape
                    @test typeof(geteval(e)) == eltype(evalspace(e))
                end

                @test let e = makeenv(), s = getstate(e), a = getaction(e), o = getobs(e)
                    r1 = geteval(s, a, o, e)
                    r2 = geteval(e)
                    randreset!(e)
                    r3 = geteval(s, a, o, e)
                    r1 == r2 == r3
                end
            end

            @testset "isdone" begin
                @test let e = makeenv()
                    isdone(e) isa Bool
                end

                @test let e = makeenv(), s = getstate(e), a = getaction(e), o = getobs(e)
                    d1 = isdone(s, a, o, e)
                    d2 = isdone(e)
                    randreset!(e)
                    d3 = isdone(s, a, o, e)
                    d1 == d2 == d3
                end
            end

            @testset "constructor consistency" begin
                e1, e2 = makeenv(), makeenv()
                @test getstate(e1) == getstate(e2)
                @test getaction(e1) == getaction(e2)
                @test getobs(e1) == getobs(e2)
                @test getreward(e1) == getreward(e2)
                @test geteval(e1) == geteval(e2)
            end

            @testset "reset" begin
                let e = makeenv()
                    @test e === reset!(e)
                end

                let e1 = makeenv(), e2 = makeenv()
                    reset!(e2)
                    @test getstate(e1) == getstate(e2)
                    @test getaction(e1) == getaction(e2)
                    @test getobs(e1) == getobs(e2)
                    @test getreward(e1) == getreward(e2)
                    @test geteval(e1) == geteval(e2)
                end
            end

            @testset "randreset" begin
                let e = makeenv()
                    @test e === randreset!(e)
                    @test e === randreset!(Random.default_rng(), e)
                end

                let e1 = makeenv(), e2 = makeenv(), rng = Random.MersenneTwister()
                    Random.seed!(rng, 1)
                    randreset!(rng, e1)

                    @test getstate(e1) != getstate(e2)
                    @test getaction(e1) == getaction(e2)
                    @test getobs(e1) != getobs(e2)
                    # TODO For some environments, resetting state may not yield
                    # different reward/eval
                    # @test getreward(e1) != getreward(e2)
                    # @test geteval(e1) != geteval(e2)
                end

                let e1 = makeenv(), e2 = makeenv(), rng = Random.MersenneTwister()
                    Random.seed!(rng, 1)
                    randreset!(rng, e1)
                    Random.seed!(rng, 1)
                    randreset!(rng, e2)

                    @test getstate(e1) == getstate(e2)
                    @test getaction(e1) == getaction(e2)
                    @test getobs(e1) == getobs(e2)
                    # TODO For some environments, resetting state may not yield
                    # different reward/eval
                    # @test getreward(e1) == getreward(e2)
                    # @test geteval(e1) == geteval(e2)
                end
            end

        end

        @testset "Determinism" begin
            # execute a random control _trajectory and check for repeatability
            let e1 = makeenv(), e2 = makeenv()
                actions = rand(actionspace(e1), 1000)
                t1 = _rollout(e1, actions)
                t2 = _rollout(e2, actions)
                reset!(e1)
                t3 = _rollout(e1, actions)
                @test t1 == t2 == t3
            end
        end

        @testset "Allocations" begin
            e = makeenv()
            s, a, o = getstate(e), getaction(e), getobs(e)

            @test @noalloc statespace($e)
            @test @noalloc getstate!($s, $e)
            @test @noalloc setstate!($e, $s)

            @test @noalloc actionspace($e)
            @test @noalloc getaction!($a, $e)
            @test @noalloc setaction!($e, $a)

            @test @noalloc obsspace($e)
            @test @noalloc getobs!($o, $e)

            @test @noalloc rewardspace($e)
            @test @noalloc getreward($s, $a, $o, $e)

            @test @noalloc evalspace($e)
            @test @noalloc geteval($s, $a, $o, $e)

            @test @noalloc reset!($e)
            @test @noalloc randreset!($e)
            @test @noalloc step!($e)
            @test @noalloc isdone($s, $a, $o, $e)
            @test @noalloc time($e)
            @test @noalloc timestep($e)
        end

        @testset "Type stability" begin
            e = makeenv()
            s, a, o = getstate(e), getaction(e), getobs(e)

            @inferred statespace(e)
            @inferred getstate!(s, e)
            @inferred setstate!(e, s)

            @inferred actionspace(e)
            @inferred getaction!(a, e)
            @inferred setaction!(e, a)

            @inferred obsspace(e)
            @inferred getobs!(o, e)

            @inferred rewardspace(e)
            @inferred getreward(s, a, o, e)

            @inferred evalspace(e)
            @inferred geteval(s, a, o, e)

            @inferred reset!(e)
            @inferred randreset!(e)
            @inferred step!(e)
            @inferred isdone(s, a, o, e)
            @inferred time(e)
            @inferred timestep(e)
        end
    end
end