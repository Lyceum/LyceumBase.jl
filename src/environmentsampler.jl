struct EnvironmentSampler{N,E<:AbstractEnvironment,B<:TrajectoryVector}
    environments::NTuple{N,E}
    buffers::NTuple{N,B}
    function EnvironmentSampler(env_tconstructor)
        N = Threads.nthreads()
        envs = Tuple(env_tconstructor(N))
        bufs = ntuple(_ -> TrajectoryVector(first(envs)), N)
        new{N,eltype(envs),eltype(bufs)}(envs, bufs)
    end
end


function sample(policy!, sampler::EnvironmentSampler, ntimesteps::Integer; kwargs...)
    sample!(TrajectoryVector(first(sampler.envs)), policy!, sampler, ntimesteps; kwargs...)
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

    #if nthreads == 1
    # short circuit
    _sample(sampler, policy!, reset!, ntimesteps, Hmax)
    #else
    #    _threaded_sample(sampler, policy!, reset!, ntimesteps, Hmax)
    #    #atomic_count = Threads.Atomic{Int}(0)
    #    #@sync for _ = 1:nthreads
    #    #    Threads.@spawn _threadsample!(
    #    #        sampler,
    #    #        policy!,
    #    #        reset!,
    #    #        ntimesteps,
    #    #        Hmax,
    #    #        atomic_count,
    #    #    )
    #    #end
    #end
    # TODO
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

#function _threaded_sample(sampler::EnvironmentSampler{N2}, policy!, reset!, ntimesteps, Hmax) where {N2}
#    @assert N2 > 1
#    #N = N2 - 1
#    N = 3

#    #ntraj_requested = ntuple(_ -> Atomic{Int}(0), Val(N))
#    ntraj_requested = [Atomic{Int}(0) for _ =1:N]
#    alive = Atomic{Bool}(true)
#    ntimesteps_accumulated = 0
#    avg_horizon = Hmax

#    try
#        for tid = 1:N
#            # TODO spawnat?
#            Threads.@spawn _thread_worker(sampler, policy!, reset!, Hmax, ntraj_requested[tid], alive)
#        end

#        while ntimesteps_accumulated < ntimesteps
#            togo = ntimesteps - ntimesteps_accumulated
#            @warn "AVGHORZION: $avg_horizon"
#            estimated_ntraj_togo = trunc(Int, cld(togo, avg_horizon))
#            @info "est: $estimated_ntraj_togo"
#            ntraj_per_thread = map(length, splitrange(estimated_ntraj_togo, N))
#            @info "per thread: $ntraj_per_thread"
#            for tid = 1:length(ntraj_per_thread)
#                atomic_add!(ntraj_requested[tid], ntraj_per_thread[tid])
#            end


#            # Do my rollouts here

#            while any(req -> req[] > 0, ntraj_requested)
#                @info "wait: $ntraj_requested"
#                sleep(0.5)
#            end

#            @info "lengths: $(map(length, sampler.buffers))"
#            ntimesteps_accumulated = sum(length, sampler.buffers)
#            avg_horizon = ntimesteps_accumulated / sum(ntrajectories, sampler.buffers)
#        end
#    finally
#        alive[] = false
#    end
#end

#function _thread_worker(sampler, policy!, reset!, Hmax, ntraj_requested::Atomic{Int}, alive::Atomic{Bool})
#    tid = Threads.threadid()
#    env = sampler.environments[tid]
#    buf = sampler.buffers[tid]
#    while alive[]
#        req = ntraj_requested[]
#        @info "Thread $tid doing $req"
#        for _=1:req
#            _sample_trajectory!(buf, env, policy!, reset!, Hmax)
#        end
#        @info "Thread $tid done $req $ntraj_requested"
#        atomic_sub!(ntraj_requested, req)
#        sleep(0.2)
#    end
#end
