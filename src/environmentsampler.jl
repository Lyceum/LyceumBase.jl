struct EnvironmentSampler{E<:AbstractEnvironment,B<:TrajectoryBuffer}
    environments::Vector{E}
    buffers::Vector{B}
    function EnvironmentSampler(env_tconstructor)
        nt = Threads.nthreads()
        envs = [env_tconstructor(nt)...]
        bufs = [TrajectoryBuffer(first(envs)) for _=1:nt]
        new{eltype(envs),eltype(bufs)}(envs, bufs)
    end
end


function sample(policy!, sampler::EnvironmentSampler, nsamples::Integer; kwargs...)
    sample!(TrajectoryBuffer(first(sampler.environments)), policy!, sampler, nsamples; kwargs...)
end

function sample!(
    B::TrajectoryBuffer,
    policy!,
    sampler::EnvironmentSampler,
    nsamples::Int;
    reset! = randreset!,
    Hmax::Integer = nsamples,
    nthreads::Integer = Threads.nthreads(),
)
    0 < nsamples || throw(ArgumentError("nsamples must be > 0"))
    0 < Hmax <= nsamples || throw(ArgumentError("Hmax must be in range (0, nsamples]"))
    if !(0 < nthreads <= Threads.nthreads())
        throw(ArgumentError("nthreads must be in range (0, Threads.nthreads()]"))
    end

    foreach(empty!, sampler.buffers)

    if nthreads == 1 # short circuit to avoid threading overhead
        _sample(sampler, policy!, reset!, nsamples, Hmax)
    else
        _threaded_sample(sampler, policy!, reset!, nsamples, Hmax, nthreads)
    end

    return collate!(B, sampler.buffers, nsamples)
end

function _sample(
    sampler::EnvironmentSampler,
    policy!,
    reset!::R,
    n::Integer,
    Hmax::Integer,
) where {R}
    tid = Threads.threadid()
    env = sampler.environments[tid]
    B = sampler.buffers[tid]

    togo = n
    while togo > 0
        reset!(env)
        rollout!(policy!, B, env, min(Hmax, togo))
        togo = n - nsamples(B)
    end
    return nothing
end

function _threaded_sample(sampler::EnvironmentSampler, policy!, reset!::R, n::Integer, Hmax::Integer, nthreads::Integer) where {R}
    barriers = [Atomic{Bool}(true) for _=1:nthreads-1]
    alive = Atomic{Bool}(true)
    H = Atomic{Int}(Hmax)

    tid = Threads.threadid()
    env = sampler.environments[tid]
    B = sampler.buffers[tid]
    try
        for i=1:nthreads-1
            Threads.@spawn _thread_worker(policy!, reset!, sampler, H, alive, barriers[i])
        end

        togo = n
        while togo > 0
            H[] = min(Hmax, togo)
            # Start rollouts in other threads
            foreach(b -> b[] = true, barriers)
            # Do this thread's rollout
            reset!(env)
            rollout!(policy!, B, env, H[])
            # Wait for each thread to finish
            while any(b -> b[], barriers) end
            togo = n - sum(nsamples, sampler.buffers)
        end
    finally
        alive[] = false
    end
    return nothing
end

@inline function _thread_worker(policy!, reset!::R, sampler::EnvironmentSampler, Hmax::Atomic{Int}, alive::Atomic{Bool}, sync::Atomic{Bool}) where {R}
    tid = Threads.threadid()
    env = sampler.environments[tid]
    B = sampler.buffers[tid]
    while alive[]
        if sync[]
            reset!(env)
            rollout!(policy!, B, env, Hmax[])
            sync[] = false
        end
    end
    return nothing
end