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
    ntimesteps = 200
    env_kwargs = (max_length = 50, reward_scale = 5)
    sampler = EnvironmentSampler(n -> ntuple(i -> ToyEnv(;env_kwargs...), n))

    i = 1
    V = sample(sampler, ntimesteps, reset! = reset!, nthreads = 1) do a, s, o
        a .= i
        i += 1
    end

    @test length(V) == ntimesteps / env_kwargs.max_length
    @test all(τ -> length(τ) == env_kwargs.max_length, V)

    U = map(unwraptraj, V)

    @test all(U) do τ
        τ.S == [0, cumsum(τ.A[1:end-1])...]
    end
    @test all(U) do τ
        τ.O == 0:length(τ)-1
    end
    @test let start = 1
        all(U) do τ
            pass = τ.A == start:start+length(τ)-1
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

@testset "multi-threaded" begin

end

end # module

    #ntimesteps = 200
    #e = TestEnv(max_length = 25)
    #sampler = EnvironmentSampler(n -> ntuple(_ -> TestEnv(), n))
    #tcounts = zeros(Threads.nthreads())
    #V = New.sample(sampler, ntimesteps, Hmax=Hmax, reset! = reset!) do a, s, o
    #    tcounts[Threads.threadid()] += 1
    #    a .= Threads.threadid()
    #    return a
    #end