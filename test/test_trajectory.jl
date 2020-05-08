module TestTrajectory

include("preamble.jl")
include("toyenv.jl")
using LyceumBase: asvec, nsamples, collate!, finish

function makedata(e::AbstractEnvironment, lengths::Vector{Int})
    τs = [
        Trajectory(
            [rand(statespace(e)) for _ = 1:lengths[i]],
            [rand(observationspace(e)) for _ = 1:lengths[i]],
            [rand(actionspace(e)) for _ = 1:lengths[i]],
            rand(lengths[i]),
            rand(statespace(e)),
            rand(observationspace(e)),
            isodd(i),
        ) for i = 1:length(lengths)
    ]

    Bs = map(τs) do τ
        B = TrajectoryBuffer(e)
        nsamp = length(τ)
        resize!(B.S, nsamp)
        resize!(B.O, nsamp)
        resize!(B.A, nsamp)
        resize!(B.R, nsamp)
        resize!(B.sT, 1)
        resize!(B.oT, 1)
        resize!(B.done, 1)
        resize!(B.offsets, 2)

        B.S .= τ.S
        B.O .= τ.O
        B.A .= τ.A
        B.R .= τ.R
        B.sT[1] = τ.sT
        B.oT[1] = τ.oT
        B.done[1] = τ.done
        B.offsets[2] = length(τ)

        LyceumBase.checkrep(B)
    end

    return (Bs=Bs, τs=τs)
end

@testset "constructors" begin
    e = ToyEnv()
    @test_inferred TrajectoryBuffer(e)
    B = TrajectoryBuffer(e; sizehint = 123)
    @test length(B.S) == length(B.O) == length(B.A) == length(B.R) == 123
    @test length(B.sT) == length(B.oT) == length(B.done) == length(B.offsets) - 1 == 0
end

@testset "collate" begin
    e = ToyEnv()
    let lens = [2,3], data = makedata(e, lens)
        B = TrajectoryBuffer(e, sizehint=0)
        A = finish(collate!(B, data.Bs, sum(lens)))
        @test nsamples(B) == sum(lens)
        @test length(B) == length(lens)

        @test length(flatview(A.S)) == length(B.S) == sum(lens)
        @test length(flatview(A.O)) == length(B.O) == sum(lens)
        @test length(flatview(A.A)) == length(B.A) == sum(lens)
        @test length(flatview(A.R)) == length(B.R) == sum(lens)
        @test length(A.sT) == length(lens)
        @test length(A.oT) == length(lens)
        @test length(A.done) == length(lens)

        @test all(1:length(A)) do i
            A[i].S == data.τs[i].S
        end
        @test all(1:length(A)) do i
            A[i].O == data.τs[i].O
        end
        @test all(1:length(A)) do i
            A[i].A == data.τs[i].A
        end
        @test all(1:length(A)) do i
            A[i].R == data.τs[i].R
        end
        @test all(1:length(A)) do i
            A[i].sT == data.τs[i].sT
        end
        @test all(1:length(A)) do i
            A[i].oT == data.τs[i].oT
        end
        @test all(1:length(A)) do i
            A[i].done == data.τs[i].done
        end
        @test B.offsets == [0, cumsum(lens)...]
    end
    let lens = [0,0], data = makedata(e, lens)
        B = TrajectoryBuffer(e, sizehint=123)
        collate!(B, data.Bs, 0)
        @test nsamples(B) == sum(lens)
        @test length(B) == 0
        @test length(B.S) == 0
        @test length(B.O) == 0
        @test length(B.A) == 0
        @test length(B.R) == 0
        @test length(B.sT) == 0
        @test length(B.oT) == 0
        @test length(B.done) == 0
        @test B.offsets == [0]
    end
    let
        B = TrajectoryBuffer(e, sizehint=123)
        A = finish(collate!(B, TrajectoryBuffer[], 0))
        @test nsamples(B) == 0
        @test length(B) == 0
        @test length(B.S) == 0
        @test length(B.O) == 0
        @test length(B.A) == 0
        @test length(B.R) == 0
        @test length(B.sT) == 0
        @test length(B.oT) == 0
        @test length(B.done) == 0
        @test B.offsets == [0]
    end
    let lens = [0,0], data = makedata(e, lens)
        B = TrajectoryBuffer(e, sizehint=123)
        @test_throws ArgumentError collate!(B, data.Bs, -1)
        @test_throws ArgumentError collate!(B, data.Bs, 1)
    end
end

end # module
