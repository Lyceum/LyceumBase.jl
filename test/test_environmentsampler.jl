module TestEnvironmentSampler

include("preamble.jl")
include("toyenv.jl")
#using LyceumBase: asvec, ntimesteps, collate

function extract_tid(batch)
    u = unique(batch.actions)
    @assert length(u) == 1
    @assert length(u[1]) == 1
    tid = u[1][1]
    return tid
end

unwrap(x) = (@assert length(x) == 1; x[])
function unwraptraj(τ::Trajectory)
    Trajectory(unwrap.(τ.S), unwrap.(τ.O), unwrap.(τ.A), τ.R, τ.sT, τ.oT, τ.done)
end

@testset "single threaded" begin
    ntimesteps = 30
    env_kwargs = (max_length = 10, reward_scale = 5)
    sampler = EnvironmentSampler(n -> ntuple(i -> ToyEnv(; env_kwargs...), n))

    i = 1
    B = sample(sampler, ntimesteps, reset! = reset!, nthreads = 1) do a, o
        a .= i
        i += 1
    end

    @test length(B) == ntimesteps / env_kwargs.max_length
    @test all(τ -> length(τ) == env_kwargs.max_length, B)

    U = map(unwraptraj, B)

    @test all(U) do τ
        τ.S == [0, cumsum(τ.A[1:end-1])...]
    end
    @test all(U) do τ
        τ.O == 0:length(τ)-1
    end
    @test let start = 1
        all(U) do τ
            pass = τ.A == start:(start+length(τ)-1)
            start += length(τ)
            pass
        end
    end
    @test all(U) do τ
        τ.R == env_kwargs.reward_scale .* τ.S
    end
    @test all(U) do τ
        τ.sT[] == sum(τ.A)
    end
    @test all(U) do τ
        τ.oT[] == length(τ)
    end
    @test all(τ -> τ.done, U)
end

#@testset "multi-threaded (nthreads = $nthreads)" for nthreads in (
#    Threads.nthreads(),
#    div(Threads.nthreads(), 2),
#)
#    if Threads.nthreads() < 2
#        @warn "Cannot test EnvironmentSampler multi-threading with Threads.nthreads() < 2"
#    end

#    env_kwargs = (
#        max_length = 50,
#        reward_scale = 5,
#        step_hook = _ -> isodd(Threads.threadid()) && busyloop(0.001),
#    )
#    ntimesteps = 5 * Threads.nthreads() * env_kwargs.max_length
#    sampler = EnvironmentSampler(n -> ntuple(i -> ToyEnv(; env_kwargs...), n))

#    tcounts = zeros(Int, Threads.nthreads())
#    B = sample(sampler, ntimesteps, reset! = reset!, nthreads = nthreads) do a, o
#        a .= Threads.threadid()
#        tcounts[Threads.threadid()] += 1
#    end

#    @test length(B) == ntimesteps / env_kwargs.max_length
#    @test all(τ -> length(τ) == env_kwargs.max_length, B)

#    U = map(unwraptraj, B)

#    @test all(U) do τ
#        τ.S == [0, cumsum(τ.A[1:end-1])...]
#    end
#    @test all(U) do τ
#        τ.O == 0:length(τ)-1
#    end
#    @test all(τ -> length(unique(τ.A)) == 1, U)
#    @test all(U) do τ
#        τ.R == env_kwargs.reward_scale .* τ.S
#    end
#    @test all(U) do τ
#        τ.sT[] == sum(τ.A)
#    end
#    @test all(U) do τ
#        τ.oT[] == length(τ)
#    end
#    @test all(τ -> τ.done, U)

#    nonempty = filter(B -> length(B) > 0, sampler.buffers)
#    # test that all threads were utilized
#    @test length(nonempty) == nthreads
#    @test begin
#        minn, maxx = extrema(map(length, nonempty))
#        # test that the number of trajectories collected by each thread differs by at most 2
#        maxx - minn <= 2
#    end
#end

end # module
