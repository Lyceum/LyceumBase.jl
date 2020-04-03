struct EnvironmentSampler{E<:AbstractEnvironment,B<:TrajectoryVector}
    environments::Vector{E}
    buffers::Vector{B}
    function EnvironmentSampler(env_tconstructor)
        nt = Threads.nthreads()
        envs = [env_tconstructor(nt)...]
        bufs = [TrajectoryVector(first(envs)) for _=1:nt]
        new{eltype(envs),eltype(bufs)}(envs, bufs)
    end
end


function sample(policy!, sampler::EnvironmentSampler, ntimesteps::Integer; kwargs...)
    sample!(TrajectoryVector(first(sampler.environments)), policy!, sampler, ntimesteps; kwargs...)
end

function sample!(
    τ::TrajectoryVector,
    policy!,
    sampler::EnvironmentSampler,
    ntimesteps::Int;
    reset! = randreset!,
    Hmax::Integer = ntimesteps,
    nthreads::Integer = Threads.nthreads(), # TODO reset
)
    0 < ntimesteps || throw(ArgumentError("ntimesteps must be > 0"))
    0 < Hmax <= ntimesteps || throw(ArgumentError("Hmax must be in range (0, ntimesteps]"))
    if !(0 < nthreads <= Threads.nthreads())
        throw(ArgumentError("`nthreads` must be in range (0, Threads.nthreads()]"))
    end

    foreach(empty!, sampler.buffers)

    if nthreads == 1
        #short circuit
        _sample(sampler, policy!, reset!, ntimesteps, Hmax)
    else
        _threaded_sample(sampler, policy!, reset!, ntimesteps, Hmax, nthreads)
    end

    return collate!(τ, sampler.buffers, first(sampler.environments), ntimesteps)
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
    buf = sampler.buffers[tid]

    togo = n - ntimesteps(buf)
    while togo > 0
        reset!(env)
        rollout!(policy!, buf, env, min(Hmax, togo))
        togo = n - ntimesteps(buf)
    end
    return nothing
end

function _threaded_sample(sampler::EnvironmentSampler, policy!, reset!, n, Hmax, nthreads)
    ntraj_requested = [Atomic{Int}(0) for _ =1:nthreads-1]
    alive = Atomic{Bool}(true)
    ntimesteps_accumulated = 0
    avg_horizon = Hmax

    try
        for tid = 1:nthreads-1
            e = sampler.environments[tid]
            b = sampler.buffers[tid]
            req = ntraj_requested[tid]
            Threads.@spawn _thread_worker(e, b, policy!, reset!, Hmax, req, alive)
        end

        env = sampler.environments[end]
        buf = sampler.buffers[end]

        while ntimesteps_accumulated < n
            togo = n - ntimesteps_accumulated
            #@warn "AVGHORZION: $avg_horizon"
            estimated_ntraj_togo = trunc(Int, cld(togo, avg_horizon))
            #@info "est: $estimated_ntraj_togo"
            ntraj_per_thread = map(length, splitrange(estimated_ntraj_togo, nthreads))
            #@info "per thread: $ntraj_per_thread"
            for tid = 1:length(ntraj_per_thread)-1
                atomic_add!(ntraj_requested[tid], ntraj_per_thread[tid])
            end

            # Do my rollouts here
            for _=1:ntraj_per_thread[end]
                reset!(env)
                rollout!(policy!, buf, env, Hmax)
            end

            while any(req -> req[] > 0, ntraj_requested)
                #@info "wait: $(map(x -> x[], ntraj_requested))"
                #sleep(0.2)
            end

            #@info "WOAH"
            #@info "lengths: $(map(length, sampler.buffers))"
            ntimesteps_accumulated = sum(ntimesteps, sampler.buffers)
            avg_horizon = ntimesteps_accumulated / sum(length, sampler.buffers)
        end
    finally
        alive[] = false
    end
end

function _thread_worker(env, buf, policy!, reset!, Hmax, ntraj_requested::Atomic{Int}, alive::Atomic{Bool})
    tid = Threads.threadid()
    while alive[]
        req = ntraj_requested[]
        for _=1:req
            reset!(env)
            rollout!(policy!, buf, env, Hmax)
        end
        #@info "Thread $tid done $req $ntraj_requested"
        atomic_sub!(ntraj_requested, req)
        #sleep(0.002)
    end
end
