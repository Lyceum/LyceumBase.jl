struct TrajectoryBuffer{B1<:ElasticBuffer,B2<:ElasticBuffer}
    trajectory::B1
    terminal::B2
end

function TrajectoryBuffer(env::AbstractEnvironment; sizehint::Maybe{Integer} = nothing, dtype::Maybe{DataType} = nothing)
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))

    trajectory = ElasticBuffer(
        states = sp.statespace,
        observations = sp.obsspace,
        actions = sp.actionspace,
        rewards = sp.rewardspace,
        evaluations = sp.evalspace
    )

    terminal = ElasticBuffer(
        states = sp.statespace,
        observations = sp.obsspace,
        dones = Bool,
        lengths = Int,
    )

    if !isnothing(sizehint)
        sizehint!(trajectory, sizehint)
        sizehint!(terminal, sizehint)
    end

    TrajectoryBuffer(trajectory, terminal)
end


struct EnvSampler{N, E<:AbstractEnvironment,B<:TrajectoryBuffer,BA}
    envs::NTuple{N,E}
    bufs::NTuple{N,B}
    batch::BA
    lock::ReentrantLock
    count::Base.RefValue{Int}
end

function EnvSampler(env_tconstructor; sizehint::Maybe{Integer} = nothing, dtype::Maybe{DataType} = nothing)
    envs = Tuple(env_tconstructor(Threads.nthreads()))
    bufs = ntuple(_ -> TrajectoryBuffer(first(envs), sizehint=sizehint, dtype=dtype), length(envs))
    batch = makebatch(first(envs), sizehint=sizehint, dtype=dtype)
    EnvSampler(envs, bufs, batch, ReentrantLock(), Ref(0))
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
    sampler::EnvSampler{N},
    nsamples::Integer;
    Hmax::Integer = nsamples,
    nthreads::Integer = N,
    copy::Bool = false,
) where {N}
    nsamples > 0 || error("nsamples must be > 0")
    (0 < Hmax <= nsamples) || error("Hmax must be in range (0, nsamples]")
    (0 < nthreads <= N) || error("nthreads must be in range (0, length(sampler.envs)]")

    nthreads = _max_threads(nsamples, Hmax, nthreads)
    emptybufs!(sampler)
    sampler.count[] = 0

    if nthreads == 1
        # short circuit
        _sample!(actionfn!, resetfn!, sampler, nsamples, Hmax)
    else
        @sync for tid = 1:nthreads
            Threads.@spawn _threadsample!(
                actionfn!,
                resetfn!,
                sampler,
                nsamples,
                Hmax,
                tid,
                sampler.lock,
                sampler.count
            )
        end
    end
    collate!(sampler, nsamples, copy)
    sampler.batch
end

function _sample!(actionfn!::F, resetfn!::G, sampler, nsamples, Hmax) where {F,G}
    env = first(sampler.envs)
    buf = first(sampler.bufs)
    traj = buf.trajectory
    term = buf.terminal

    resetfn!(env)
    trajlength = n = 0
    while n < nsamples
        rolloutstep!(actionfn!, traj, env)
        done = isdone(env)
        trajlength += 1

        if done || trajlength == Hmax
            terminate_trajectory!(term, env, trajlength, done)
            resetfn!(env)
            n += trajlength
            trajlength = 0
        end
    end
    nothing
end

function _threadsample!(
    actionfn!::F,
    resetfn!::G,
    sampler,
    nsamples,
    Hmax,
    tid,
    lck,
    count,
) where {F,G}
    env = sampler.envs[tid]
    buf = sampler.bufs[tid]

    resetfn!(env)
    trajlength = 0
    while true
        done = rolloutstep!(actionfn!, buf.trajectory, env)
        trajlength += 1

        lock(lck)
        if count[] >= nsamples # Another thread finished the job
            unlock(lck)
            break
        elseif count[] + trajlength == nsamples # We have exactly enough samples to finish the job
            count[] = nsamples
            unlock(lck)
            terminate_trajectory!(buf.terminal, env, done, trajlength)
            break
        elseif count[] + trajlength > nsamples # We have more samples than needed to finish the job
            trajlength = nsamples - count[]
            count[] = nsamples
            unlock(lck)
            done = false # since we threw away at least the last sample, we didn't actually terminate early
            terminate_trajectory!(buf.terminal, env, done, trajlength)
            break
        elseif done || trajlength == Hmax # We should terminate this trajectory and keep sampling
            count[] += trajlength
            unlock(lck)
            terminate_trajectory!(buf.terminal, env, done, trajlength)
            resetfn!(env)
            trajlength = 0
        else
            # Just keep sampling
            unlock(lck)
        end
    end
    nothing
end

function rolloutstep!(actionfn!::F, traj::ElasticBuffer, env::AbstractEnvironment) where {F}
    grow!(traj) # extend our buffer by one to accomdate the next sample
    t = lastindex(traj)
    @uviews traj begin
        @inbounds st = view(traj.states, :, t)
        @inbounds at = view(traj.actions, :, t)
        @inbounds ot = view(traj.observations, :, t)

        getstate!(st, env)
        getobs!(ot, env)

        actionfn!(at, st, ot)
        setaction!(env, at)
        step!(env)

        r = getreward(st, at, ot, env)
        e = geteval(st, at, ot, env)
        done = isdone(st, at, ot, env)

        @inbounds traj.rewards[t] = r
        @inbounds traj.evaluations[t] = e
        return done
    end
end

function terminate_trajectory!(term::ElasticBuffer, env::AbstractEnvironment, done::Bool, trajlength::Int)
    grow!(term) # extend our terminal buffer by one
    i = lastindex(term)
    @uviews term begin
        thisterm = view(term, i)
        getstate!(vec(thisterm.states), env) # TODO
        getobs!(vec(thisterm.observations), env) # TODO
        term.dones[i] = done
        term.lengths[i] = trajlength
    end
    term
end

# allocate a batch for env's spaces
function makebatch(env::AbstractEnvironment; sizehint::Union{Integer,Nothing} = nothing, dtype::Maybe{DataType} = nothing)
    # optionally override storage type of env's spaces to be dtype
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))
    batch = (
        states = BatchedArray(sp.statespace),
        observations = BatchedArray(sp.obsspace),
        actions = BatchedArray(sp.actionspace),
        rewards = BatchedArray(sp.rewardspace),
        evaluations = BatchedArray(sp.evalspace),
        terminal_states = BatchedArray(sp.statespace),
        terminal_observations = BatchedArray(sp.obsspace),
        dones = Vector{Bool}(),
    )
    sizehint !== nothing && foreach(el -> sizehint!(el, sizehint), batch)
    batch
end



function collate!(sampler::EnvSampler, N::Integer, copy::Bool)
    batch = copy ? map(deepcopy, sampler.batch) : sampler.batch # TODO
    for b in batch
        empty!(b)
        sizehint!(b, N)
    end

    for buf in sampler.bufs

        trajbuf = buf.trajectory
        termbuf = buf.terminal
        from = firstindex(trajbuf)

        @uviews trajbuf termbuf for episode_idx in eachindex(termbuf)

            len = termbuf.lengths[episode_idx]
            until = from + len
            to = until - 1

            traj = view(trajbuf, from:to)
            term = view(termbuf, episode_idx)

            push!(batch.states, traj.states)
            push!(batch.observations, traj.observations)
            push!(batch.actions, traj.actions)
            push!(batch.rewards, traj.rewards)
            push!(batch.evaluations, traj.evaluations)
            push!(batch.terminal_states, reshape(term.states, (:, 1))) # TODO (:, 1)
            push!(batch.terminal_observations, reshape(term.observations, (:, 1))) #TODO (:, 1)
            push!(batch.dones, termbuf.dones[episode_idx])

            from = until
        end
    end

    nbatches = length(batch.dones)
    @assert length(batch.terminal_states) ==
            length(batch.terminal_observations) ==
            length(batch.dones)
    @assert nsamples(batch.states) == N
    @assert nsamples(batch.observations) == N
    @assert nsamples(batch.actions) == N
    @assert nsamples(batch.rewards) == N
    @assert nsamples(batch.evaluations) == N

    batch
end


function sortlongest(sampler, N)
    bufs = sampler.bufs
    allepisode_idxs = NTuple{3, Int}[]
    count = 0
    for (buf_idx, buf) in enumerate(bufs), (episode_idx, len) in enumerate(buf.terminal.lengths)
        count >= N && break
        push!(allepisode_idxs, (len,buf_idx,episode_idx))
        count += len
    end
    sort!(allepisode_idxs, rev=true)
end


function _max_threads(nsamples, Hmax, nthreads)
    d, r = divrem(nsamples, Hmax)
    r > 0 && (d += 1)
    min(d, nthreads)
end