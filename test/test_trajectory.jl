module TestTrajectory

include("preamble.jl")
include("toyenv.jl")
using LyceumBase: asvec, ntimesteps, collate

function makedata(e::AbstractEnvironment, lengths::Vector{Int}; isdone = isodd)
    τs = [
        Trajectory(
            [rand(statespace(e)) for _ = 1:lengths[i]],
            [rand(obsspace(e)) for _ = 1:lengths[i]],
            [rand(actionspace(e)) for _ = 1:lengths[i]],
            rand(lengths[i]),
            rand(statespace(e)),
            rand(obsspace(e)),
            isdone(i),
        ) for i = 1:length(lengths)
    ]
end

@testset "constructors" begin
    e = ToyEnv()
    @test_inferred TrajectoryVector(e)
    V = TrajectoryVector(e; sizehint = 123)
    @test length(V.S) == length(V.O) == length(V.A) == length(V.R) == 123
    @test length(V.sT) == length(V.oT) == length(V.done) == length(V.offsets) - 1 == 0
end

@testset "indexing" begin
    V = TrajectoryVector(rand(6), rand(6), rand(6), rand(6), rand(3), rand(3), [true, false, true], [0, 1, 3, 6], 3)
    @test length(V) == 3
    @test ntimesteps(V) == 6
    parent(V.S)[:] .= 1:6
    parent(V.O)[:] .= 2:7
    parent(V.A)[:] .= 3:8
    parent(V.R)[:] .= 4:9
    @test V[1].S == [1]
    @test V[1].O == [2]
    @test V[1].A == [3]
    @test V[1].R == [4]

    @test V[2].S == [2, 3]
    @test V[2].O == [3, 4]
    @test V[2].A == [4, 5]
    @test V[2].R == [5, 6]

    @test V[3].S == [4, 5, 6]
    @test V[3].O == [5, 6, 7]
    @test V[3].A == [6, 7, 8]
    @test V[3].R == [7, 8, 9]
end

@testset "push!/append!/empty!" begin
    e = ToyEnv()
    ssp = statespace(e)
    osp = obsspace(e)
    asp = actionspace(e)
    V = TrajectoryVector(e)
    @test length(V) == 0
    τs = makedata(e, [2, 4, 6, 8])

    append!(V, τs[1:2])
    append!(V, τs[3:4])
    @test length(V) == 4
    @test ntimesteps(V) == sum(length, τs)
    @test V == τs

    empty!(V)
    @test length(V) == 0
    @test ntimesteps(V) == 0

    append!(V, τs[1:2])
    append!(V, τs[3:4])
    @test length(V) == 4
    @test ntimesteps(V) == sum(length, τs)
    @test V == τs

    empty!(V)
    @test length(V) == 0
    @test ntimesteps(V) == 0
    push!(V, τs[1])
    @test V[1] == τs[1]
end


@testset "collate" begin
    e = ToyEnv()

    let
        Vs = (TrajectoryVector(e), TrajectoryVector(e), TrajectoryVector(e))
        τs = makedata(e, [2, 3, 4])
        for (V, τ) in zip(Vs, τs)
            push!(V, τ)
        end
        C = collate(Vs, e, 9)
        @test_inferred collate(Vs, e, 9)
        @test ntimesteps(C) == 9
        @test C == τs
    end

    let
        Vs = (TrajectoryVector(e), TrajectoryVector(e), TrajectoryVector(e))
        τs = makedata(e, [2, 3, 4]; isdone = i -> true)
        for (V, τ) in zip(Vs, τs)
            push!(V, τ)
        end
        C = collate(Vs, e, 8)
        @test_inferred collate(Vs, e, 8)
        @test ntimesteps(C) == 8
        @test all(C.done[1:2])
        @test C[1:2] == τs[1:2]
        @test begin
            τ = Trajectory(
                τs[3].S[1:3],
                τs[3].O[1:3],
                τs[3].A[1:3],
                τs[3].R[1:3],
                τs[3].S[4],
                τs[3].O[4],
                false
            )
            C[3] == τ
        end
    end
end

end # module
