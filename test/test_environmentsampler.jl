module TestEnvironmentSampler

#@testset "basic" begin

#const NTIMESTEPS = 200
#    ntimesteps = 360
#    Hmax = 6
#    e = New.TestEnv()
#    sampler = New.EnvironmentSampler(n -> ntuple(_ -> New.TestEnv(), n))
#    tcounts = zeros(Threads.nthreads())
#    batches = New.sample(sampler, nsamp, Hmax=Hmax, reset! = reset!) do a, s, o
#        tcounts[Threads.threadid()] += 1
#        a .= Threads.threadid()
#        return a
#    end

#    # TODO make StructArray return Trajectory instead of NT so that
#    # it has a length?
#    for b in batches
#        tid = extract_tid(b)
#        trajlength = length(b.states)
#        @assert all(1:trajlength) do i
#            length(b.states[i]) == 1 && first(b.states[i]) == tid * (i - 1)
#        end
#        @assert all(1:trajlength) do i
#            length(b.observations[i]) == 1 && first(b.observations[i]) == first(b.states[i]) * New.OBSERVATION_SCALE
#        end
#        @assert all(1:trajlength) do i
#            length(b.rewards[i]) == 1 && first(b.rewards[i]) == first(b.states[i]) * New.REWARD_SCALE
#        end
#    end

#    tc = zeros(Int, Threads.nthreads())
#    for b in batches
#        tc[extract_tid(b)] += 1
#    end
#    @info "counts: $tc"

end # module
