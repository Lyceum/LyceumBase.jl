module TestTrajectory

include("preamble.jl")
include("toyenv.jl")
using LyceumBase: asvec, ntimesteps, collate


@testset "constructors" begin
    e = ToyEnv()
    @test_inferred TrajectoryVector(e)
    V = TrajectoryVector(e; sizehint = 123)
    @test length(V.S) == length(V.O) == length(V.A) == length(V.R) == 123
    @test length(V.done) == length(V.offsets) - 1 == 0
end

@testset "indexing" begin
    V = TrajectoryVector(rand(9), rand(9), rand(6), rand(6), [true, false, true], [0, 1, 3, 6], 3)
    @test length(V) == 3
    @test ntimesteps(V) == 6
    parent(V.S)[:] .= 1:9
    parent(V.O)[:] .= 2:10
    parent(V.A)[:] .= 1:6
    parent(V.R)[:] .= 2:7
    @test V[1].S == [1, 2]
    @test V[1].O == [2, 3]
    @test V[1].A == [1]
    @test V[1].R == [2]

    @test V[2].S == [3, 4, 5]
    @test V[2].O == [4, 5, 6]
    @test V[2].A == [2, 3]
    @test V[2].R == [3, 4]

    @test V[3].S == [6, 7, 8, 9]
    @test V[3].O == [7, 8, 9, 10]
    @test V[3].A == [4, 5, 6]
    @test V[3].R == [5, 6, 7]
end

@testset "push!/append!/empty!" begin
    e = ToyEnv()
    ssp = statespace(e)
    osp = obsspace(e)
    asp = actionspace(e)
    V = TrajectoryVector(e)
    @test length(V) == 0
    τs = [
        Trajectory([rand(ssp), rand(ssp)], [rand(osp), rand(osp)], [rand(asp)], rand(1), isodd(i)) for i = 1:10
    ]

    append!(V, τs[1:5])
    append!(V, τs[6:10])
    @test length(V) == 10
    @test V == τs

    empty!(V)
    @test length(V) == 0

    append!(V, τs[1:5])
    append!(V, τs[6:10])
    @test length(V) == 10
    @test V == τs

    empty!(V)
    @test length(V) == 0
    push!(V, τs[1])
    @test V[1] == τs[1]
end

function makedata(e::AbstractEnvironment, lengths::Vector{Int}; isdone = isodd)
    τs = [
        Trajectory(
            [rand(statespace(e)) for _ = 1:lengths[i]+1],
            [rand(obsspace(e)) for _ = 1:lengths[i]+1],
            [rand(actionspace(e)) for _ = 1:lengths[i]],
            rand(lengths[i]),
            isdone(i),
        ) for i = 1:length(lengths)
    ]
end

@testset "collate" begin
    e = ToyEnv()
    ssp = statespace(e)
    osp = obsspace(e)
    asp = actionspace(e)

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
                τs[3].S[1:4],
                τs[3].O[1:4],
                τs[3].A[1:3],
                τs[3].R[1:3],
                false
            )
            C[3] == τ
        end
    end
end

end # module
