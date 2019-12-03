struct TrajectoryBuffer{B1<:ElasticBuffer,B2<:ElasticBuffer}
    trajectory::B1
    terminal::B2
end

function TrajectoryBuffer(env::AbstractEnv; sizehint::Union{Integer,Nothing} = nothing, dtype::Maybe{DataType} = nothing)
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))

    trajectory = ElasticBuffer(
        states = sp.statespace,
        observations = sp.observationspace,
        actions = sp.actionspace,
        rewards = sp.rewardspace,
        evaluations = sp.evaluationspace
    )

    terminal = ElasticBuffer(
        states = sp.statespace,
        observations = sp.observationspace,
        dones = Bool,
        lengths = Int,
    )

    if !isnothing(sizehint)
        sizehint!(trajectory, sizehint)
        sizehint!(terminal, sizehint)
    end

    TrajectoryBuffer(trajectory, terminal)
end


struct EnvSampler{E<:AbstractEnv,B<:TrajectoryBuffer,BA}
    envs::Vector{E}
    bufs::Vector{B}
    batch::BA
end

function EnvSampler(env_ctor::Function; sizehint::Union{Integer,Nothing} = nothing, dtype::Maybe{DataType} = nothing)
    envs = [e for e in env_ctor(Threads.nthreads())]
    bufs = [TrajectoryBuffer(first(envs), sizehint=sizehint, dtype=dtype) for _ = 1:Threads.nthreads()]
    batch = makebatch(first(envs), sizehint=sizehint, dtype=dtype)
    EnvSampler(envs, bufs, batch)
end

function emptybufs!(sampler::EnvSampler)
    for buf in sampler.bufs
        empty!(buf.trajectory)
        empty!(buf.terminal)
    end
end


function sample!(
    actionfn!,
    resetfn!,
    sampler::EnvSampler,
    nsamples::Integer;
    Hmax::Integer = nsamples,
    nthreads::Integer = Threads.nthreads(),
    copy::Bool = false,
)
    nsamples > 0 || error("nsamples must be > 0")
    (0 < Hmax <= nsamples) || error("Hmax must be 0 < Hmax <= nsamples")
    (
        0 < nthreads <= Threads.nthreads()
    ) || error("nthreads must b 0 < nthreads < Threads.nthreads()")

    atomiccount = Threads.Atomic{Int}(0)
    atomicidx = Threads.Atomic{Int}(1)

    nthreads = _defaultnthreads(nsamples, Hmax, nthreads)
    emptybufs!(sampler)

    if nthreads == 1
        _threadsample!(actionfn!, resetfn!, sampler, nsamples, Hmax)
    else
        @sync for _ = 1:nthreads
            Threads.@spawn _threadsample!(
                actionfn!,
                resetfn!,
                sampler,
                nsamples,
                Hmax,
                atomiccount,
            )
        end
    end

    @assert sum(b -> length(b.trajectory), sampler.bufs) >= nsamples

    collate!(sampler, nsamples, copy)
    sampler.batch
end

function _threadsample!(actionfn!::F, resetfn!::G, sampler, nsamples, Hmax) where {F,G}
    env = sampler.envs[Threads.threadid()]
    buf = sampler.bufs[Threads.threadid()]
    traj = buf.trajectory
    term = buf.terminal

    resetfn!(env)
    trajlength = n = 0
    while n < nsamples
        rolloutstep!(actionfn!, traj, env)
        done = isdone(env)
        trajlength += 1

        if done || trajlength == Hmax
            terminate!(term, env, trajlength, done)
            resetfn!(env)
            n += trajlength
            trajlength = 0
        end
    end
    sampler
end




function _threadsample!(
    actionfn!::F,
    resetfn!::G,
    sampler,
    nsamples,
    Hmax,
    atomiccount,
) where {F,G}
    env = sampler.envs[Threads.threadid()]
    buf = sampler.bufs[Threads.threadid()]
    traj = buf.trajectory
    term = buf.terminal

    resetfn!(env)
    trajlength = 0
    while true
        done = rolloutstep!(actionfn!, traj, env)
        trajlength += 1

        if atomiccount[] >= nsamples
            break
        elseif atomiccount[] + trajlength >= nsamples
            Threads.atomic_add!(atomiccount, trajlength)
            terminate!(term, env, trajlength, done)
            break
        elseif done || trajlength == Hmax
            Threads.atomic_add!(atomiccount, trajlength)
            terminate!(term, env, trajlength, done)
            resetfn!(env)
            trajlength = 0
        end
    end
    sampler
end


function rolloutstep!(actionfn!::F, traj::ElasticBuffer, env::AbstractEnv) where {F}
    grow!(traj)
    t = lastindex(traj)
    @uviews traj begin
        st, ot, at =
            view(traj.states, :, t), view(traj.observations, :, t), view(traj.actions, :, t)
        getstate!(st, env)
        getobs!(ot, env)

        actionfn!(at, st, ot)
        #setaction!(env, at)
        r, e, done = step!(env, at)
        #step!(env)

        traj.rewards[t] = r
        traj.evaluations[t] = e
        return done

        #traj.rewards[t] = getreward(env)
        #traj.evaluations[t] = geteval(env)
    end
end

function terminate!(term::ElasticBuffer, env::AbstractEnv, trajlength::Integer, done::Bool)
    grow!(term)
    i = lastindex(term)
    @uviews term begin
        thistraj = view(term, i)
        getstate!(vec(thistraj.states), env) # TODO
        getobs!(vec(thistraj.observations), env) # TODO
        term.dones[i] = done
        term.lengths[i] = trajlength
    end
    term
end


function makebatch(env::AbstractEnv; sizehint::Union{Integer,Nothing} = nothing, dtype::Maybe{DataType} = nothing)
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))
    batch = (
        states = BatchedArray(sp.statespace),
        observations = BatchedArray(sp.observationspace),
        actions = BatchedArray(sp.actionspace),
        rewards = BatchedArray(sp.rewardspace),
        evaluations = BatchedArray(sp.evaluationspace),
        terminal_states = BatchedArray(sp.statespace),
        terminal_observations = BatchedArray(sp.observationspace),
        dones = Vector{Bool}(),
    )
    !isnothing(sizehint) && foreach(el -> sizehint!(el, sizehint), batch)
    batch
end

function collate!(sampler::EnvSampler, n::Integer, copy::Bool)
    batch = copy ? map(deepcopy, sampler.batch) : sampler.batch
    for b in batch
        empty!(b)
        sizehint!(b, n)
    end

    count = 0
    maxlen = maximum(map(buf -> length(buf.terminal), sampler.bufs))
    for i = 1:maxlen, (tid, buf) in enumerate(sampler.bufs)
        count >= n && break

        trajbuf = buf.trajectory
        termbuf = buf.terminal
        length(termbuf) < i && continue

        @uviews trajbuf termbuf begin
            from = i == 1 ? 1 : sum(view(termbuf.lengths, 1:i-1)) + 1
            len = termbuf.lengths[i]
            len = count + len > n ? n - count : len
            to = from + len - 1

            traj = view(trajbuf, from:to)
            term = view(termbuf, i)

            push!(batch.states, traj.states)
            push!(batch.observations, traj.observations)
            push!(batch.actions, traj.actions)
            push!(batch.rewards, traj.rewards)
            push!(batch.evaluations, traj.evaluations)
            push!(batch.terminal_states, reshape(term.states, (:, 1)))
            push!(batch.terminal_observations, reshape(term.observations, (:, 1)))
            push!(batch.dones, termbuf.dones[i])

            count += length(from:to)
            from = to + 1
        end
    end

    nbatches = length(batch.dones)
    @assert length(batch.terminal_states) ==
            length(batch.terminal_observations) ==
            length(batch.dones)
    @assert nsamples(batch.states) == n
    @assert nsamples(batch.observations) == n
    @assert nsamples(batch.actions) == n
    @assert nsamples(batch.rewards) == n
    @assert nsamples(batch.evaluations) == n


    batch
end


function _defaultnthreads(nsamples, Hmax, nthreads)
    d, r = divrem(nsamples, Hmax)
    if r > 0
        d += 1
    end
    min(d, nthreads)
end










export NaieveEnvSampler
struct NaieveEnvSampler{E<:AbstractEnv,B1,B2}
    envs::Vector{E}
    trajectory::B1
    terminal::B2
end

function NaieveEnvSampler(env_ctor::Function, K::Integer, H::Integer)
    envs = [e for e in env_ctor(Threads.nthreads())]
    ssp, osp, asp, rsp, esp = spaces(first(envs))
    trajectory = (
        states = allocate(ssp, H, K),
        observations = allocate(osp, H, K),
        actions = allocate(asp, H, K),
        rewards = allocate(rsp, H, K),
        evaluations = allocate(esp, H, K),
    )
    terminal = (states = allocate(ssp, K), observations = allocate(osp, K))
    NaieveEnvSampler(envs, trajectory, terminal)
end

function rollout!(env, actionfn!, resetfn!, sk, ok, ak, rk, ek, termstatek, termobsk)
    T = length(rk)
    #resetfn!(env)
    #randreset!(env)
    reset!(env)
    @views for t = 1:T
        st, ot, at = view(sk, :, t), view(ok, :, t), view(ak, :, t)

        #getstate!(st, env)
        #getobs!(ot, env)
        #actionfn!(at, st, ot)
        #step!(env, at)

        getstate!(sk[:, t], env)
        getobs!(ok[:, t], env)

        actionfn!(ak[:, t], sk[:, t], ok[:, t])
        setaction!(env, ak[:, t])
        step!(env)

        rk[t] = getreward(env)
        ek[t] = geteval(env)
    end
    getstate!(termstatek, env)
    getobs!(termobsk, env)
end

function rosierollout!(
    getaction!::F,
    s::AbstractMatrix,
    o::AbstractMatrix,
    a::AbstractMatrix,
    r::AbstractVector,
    e::AbstractVector,
    terms::AbstractVector,
    termo::AbstractVector,
    env,
) where {F}

    T = size(s, 2)
    @assert size(o, 2) == size(a, 2) == length(r) == length(e) == T

    @uviews s o a for t = 1:T
        st, ot, at = view(s, :, t), view(o, :, t), view(a, :, t)
        getstate!(st, env)
        getobs!(ot, env)
        getaction!(at, st, ot)
        step!(env, at)
        #rew, eval, done = step!(env, at)
        rew = step!(env, at)
        r[t] = rew
        e[t] = rew
    end
    getstate!(terms, env)
    getobs!(termo, env)
    nothing
end
function _threadsample!(
    actionfn!,
    resetfn!,
    sampler::NaieveEnvSampler,
    range::UnitRange{Int},
)
    traj, term = sampler.trajectory, sampler.terminal
    @unpack states, observations, actions, rewards, evaluations = traj

    termstates, termobses = term.states, term.observations
    env = sampler.envs[Threads.threadid()]

    @uviews states observations actions rewards evaluations termstates termobses for k in range
        sk, ok, ak =
            view(states, :, :, k), view(observations, :, :, k), view(actions, :, :, k)
        rk, ek = view(rewards, :, k), view(evaluations, :, k)
        termstatek, termobsk = view(termstates, :, k), view(termobses, :, k)
        rollout!(env, actionfn!, resetfn!, sk, ok, ak, rk, ek, termstatek, termobsk)
        #randreset!(env)
        #rosierollout!(actionfn!, sk, ok, ak, rk, ek, termstatek, termobsk, env)
    end
end


function sample!(
    actionfn!,
    resetfn!,
    sampler::NaieveEnvSampler,
    nthreads::Integer = Threads.nthreads(),
    copy::Bool = false,
)
    T, K = Base.tail(size(sampler.trajectory.states))
    ranges = splitrange(K, nthreads)
    randn!(sampler.trajectory.actions)
    for range in ranges
        _threadsample!(actionfn!, resetfn!, sampler, range)
    end
    branges = UnitRange{Int}[]
    branges2 = UnitRange{Int}[]
    from = 1
    for i = 1:K
        to = from + T - 1
        #to = from
        push!(branges, from:to)
        push!(branges2, i:i)
        from = to + 1
    end
    x = sampler.trajectory
    y = sampler.terminal
    f(x) = reshape(x, (:, T * K))
    g(x) = dropdims(reshape(x, (:, T * K)), dims = 1)
    batch = (
        states = BatchedArray(f(x.states), branges),
        observations = BatchedArray(f(x.observations), branges),
        actions = BatchedArray(f(x.actions), branges),
        rewards = BatchedArray(g(x.rewards), branges),
        evaluations = BatchedArray(g(x.evaluations), branges),
        terminal_states = BatchedArray(y.states, branges2),
        terminal_observations = BatchedArray(y.observations, branges2),
    )
    map(deepcopy, batch)
end













#function collate!(sampler::EnvSampler, n::Integer)
#    batch = makebatch(first(sampler.envs), n)
#    count = 0
#    for buf in sampler.bufs
#        trajbuf = buf.trajectory
#        termbuf = buf.terminal
#        from = 1
#        @uviews trajbuf termbuf for i = 1:length(termbuf)
#            if count >= n break end
#
#            len = termbuf.length[i]
#            len = count + len > n ? n - count : len
#            to = from + len - 1
#
#            traj = view(trajbuf, from:to)
#            term = view(termbuf, i)
#
#            push!(batch.states, traj.states)
#            push!(batch.observations, traj.observations)
#            push!(batch.actions, traj.actions)
#            push!(batch.rewards, traj.rewards)
#            push!(batch.evaluations, traj.evaluations)
#            push!(batch.terminal_states, reshape(term.states, (:, 1)))
#            push!(batch.terminal_observations, reshape(term.observations, (:, 1)))
#            push!(batch.dones, termbuf.done[i])
#
#            count += length(from:to)
#            from = to + 1
#        end
#    end
#
#    @assert length(batch.terminal_states) == length(batch.terminal_observations) == length(batch.dones)
#    nbatches = length(batch.dones)
#    @assert nsamples(batch.states) == n
#    @assert nsamples(batch.observations) == n
#    @assert nsamples(batch.actions) == n
#    @assert nsamples(batch.rewards) == n
#    @assert nsamples(batch.evaluations) == n
#
#    batch
#end
