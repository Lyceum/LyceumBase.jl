module TestTrajectory

include("preamble.jl")
include("toyenv.jl")
using LyceumBase: asvec, nsamples, collate

function makedata(e::AbstractEnvironment, lengths::Vector{Int}; isdone = isodd)
    τs = [
        Trajectory(
            [rand(statespace(e)) for _ = 1:lengths[i]],
            [rand(observationspace(e)) for _ = 1:lengths[i]],
            [rand(actionspace(e)) for _ = 1:lengths[i]],
            rand(lengths[i]),
            rand(statespace(e)),
            rand(observationspace(e)),
            isdone(i),
        ) for i = 1:length(lengths)
    ]
end

@testset "constructors" begin
    e = ToyEnv()
    @test_inferred TrajectoryBuffer(e)
    B = TrajectoryBuffer(e; sizehint = 123)
    @test length(B.S) == length(B.O) == length(B.A) == length(B.R) == 123
    @test length(B.sT) == length(B.oT) == length(B.done) == length(B.offsets) - 1 == 0
end

@testset "indexing" begin
    B = TrajectoryBuffer(
        rand(6),
        rand(6),
        rand(6),
        rand(6),
        rand(3),
        rand(3),
        [true, false, true],
        [0, 1, 3, 6],
        3,
    )
    @test length(B) == 3
    @test nsamples(B) == 6
    parent(B.S)[:] .= 1:6
    parent(B.O)[:] .= 2:7
    parent(B.A)[:] .= 3:8
    parent(B.R)[:] .= 4:9
    @test B[1].S == [1]
    @test B[1].O == [2]
    @test B[1].A == [3]
    @test B[1].R == [4]

    @test B[2].S == [2, 3]
    @test B[2].O == [3, 4]
    @test B[2].A == [4, 5]
    @test B[2].R == [5, 6]

    @test B[3].S == [4, 5, 6]
    @test B[3].O == [5, 6, 7]
    @test B[3].A == [6, 7, 8]
    @test B[3].R == [7, 8, 9]
end

@testset "push!/append!/empty!" begin
    e = ToyEnv()
    ssp = statespace(e)
    osp = observationspace(e)
    asp = actionspace(e)
    B = TrajectoryBuffer(e)
    @test length(B) == 0
    τs = makedata(e, [2, 4, 6, 8])

    append!(B, τs[1:2])
    append!(B, τs[3:4])
    @test length(B) == 4
    @test nsamples(B) == sum(length, τs)
    @test B == τs

    empty!(B)
    @test length(B) == 0
    @test nsamples(B) == 0

    append!(B, τs[1:2])
    append!(B, τs[3:4])
    @test length(B) == 4
    @test nsamples(B) == sum(length, τs)
    @test B == τs

    empty!(B)
    @test length(B) == 0
    @test nsamples(B) == 0
    push!(B, τs[1])
    @test B[1] == τs[1]
end


@testset "collate" begin
    e = ToyEnv()
    let
        Vs = [TrajectoryBuffer(e), TrajectoryBuffer(e), TrajectoryBuffer(e)]
        τs = makedata(e, [2, 3, 4])
        for (B, τ) in zip(Vs, τs)
            push!(B, τ)
        end
        C = collate(Vs, e, 9)
        @test_inferred collate(Vs, e, 9)
        @test nsamples(C) == 9
        @test C == τs
    end
    let
        Vs = [TrajectoryBuffer(e), TrajectoryBuffer(e), TrajectoryBuffer(e)]
        τs = makedata(e, [2, 3, 4]; isdone = i -> true)
        for (B, τ) in zip(Vs, τs)
            push!(B, τ)
        end
        C = collate(Vs, e, 8)
        @test_inferred collate(Vs, e, 8)
        @test nsamples(C) == 8
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
                false,
            )
            C[3] == τ
        end
    end
end

end # module
