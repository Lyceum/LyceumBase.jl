struct EnvironmentSampler{E<:AbstractEnvironment,B<:TrajectoryBuffer}
    environments::Vector{E}
    buffers::Vector{B}
    function EnvironmentSampler(env_tconstructor; dtype::Maybe{Type} = nothing)
        nt = Threads.nthreads()
        envs = [env_tconstructor(nt)...]
        bufs = [TrajectoryBuffer(first(envs), dtype = dtype) for _ = 1:nt]
        new{eltype(envs),eltype(bufs)}(envs, bufs)
    end
end


function Distributions.sample(
    policy!,
    sampler::EnvironmentSampler,
    nsamples::Integer;
    dtype::Maybe{DataType} = nothing,
    kwargs...,
)
    sample!(
        policy!,
        TrajectoryBuffer(first(sampler.environments), dtype = dtype),
        sampler,
        nsamples;
        kwargs...,
    )
end

function Distributions.sample!(
    policy!,
    B::TrajectoryBuffer,
    sampler::EnvironmentSampler,
    n::Integer;
    reset! = randreset!,
    Hmax::Integer = nsamples,
    nthreads::Integer = Threads.nthreads(),
    truncate::Bool = true,
)
    n > 0 || argerror("n must be > 0")
    0 < Hmax <= n || argerror("Hmax must be in range (0, nsamples]")
    if !(0 < nthreads <= Threads.nthreads())
        argerror("nthreads must be in range (0, Threads.nthreads()]")
    end

    # TODO
    foreach(empty!, sampler.buffers)

    if nthreads == 1 # short circuit to avoid threading overhead
        _sample(sampler, policy!, reset!, n, Hmax)
    else
        _threaded_sample(sampler, policy!, reset!, n, Hmax, nthreads)
    end

    # TODO test
    ns = truncate ? n : sum(nsamples, sampler.buffers)
    collate!(B, sampler.buffers, ns)
    return B
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
        len = _rollout!(policy!, B, env, max(1, min(Hmax, togo)))
        togo -= len
    end
    return nothing
end

function _threaded_sample(
    sampler::EnvironmentSampler,
    policy!,
    reset!,
    n::Integer,
    Hmax::Integer,
    nthreads::Integer,
)
    alive = Atomic{Bool}(true)
    iters = [Atomic{Int}(0) for _ = 1:nthreads]
    togo = Atomic{Int}(n)

    @static if VERSION < v"1.4"
        Threads.@sync for i = 1:nthreads
            Threads.@spawn _thread_worker(
                sampler,
                policy!,
                reset!,
                togo,
                convert(Int, Hmax),
                iters,
                iters[i],
            )
        end
    else
        Threads.@sync for i = 1:nthreads
            Threads.@spawn _thread_worker(
                $sampler,
                $policy!,
                $reset!,
                $togo,
                $(convert(Int, Hmax)),
                $iters,
                $(iters[i]),
            )
        end
    end
    return nothing
end


@noinline function _thread_worker(
    sampler::EnvironmentSampler,
    policy!,
    @specialize(reset!),
    togo::Atomic{Int},
    Hmax::Int,
    iters::Vector{Atomic{Int}},
    iter::Atomic{Int},
)
    tid = Threads.threadid()
    env = sampler.environments[tid]
    B = sampler.buffers[tid]
    stopcb = @closure () -> togo[] <= 0
    t = time()
    while togo[] > 0
        if iter[] <= minimum(getindex, iters)
            atomic_add!(iter, 1)
            reset!(env)
            len = _rollout!(policy!, B, env, max(1, min(togo[], Hmax)), stopcb)
            atomic_sub!(togo, len)
            t = time()
        else
            time() - t > 5 && error("Timeout on thread $tid. Please file a bug report.")
            # See: https://github.com/JuliaLang/julia/issues/33097
            ccall(:jl_gc_safepoint, Cvoid, ())
        end
    end
    return nothing
end
