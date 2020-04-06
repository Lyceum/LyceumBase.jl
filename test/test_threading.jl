module TestThreading

include("preamble.jl")

@testset "splitrange" begin
    @test splitrange(16, 4) == [1:4, 5:8, 9:12, 13:16]
    @test splitrange(4, 16) == [1:1, 2:2, 3:3, 4:4]
    @test splitrange(16, 5) == [1:4, 5:7, 8:10, 11:13, 14:16]
end

@testset "tseed!" begin
    rngs = [MersenneTwister() for _ = 1:Threads.nthreads()]
    x = zeros(1000)
    y = zeros(1000)
    tseed!(rngs, 1)
    Threads.@threads for i = 1:length(x)
        x[i] = rand(rngs[Threads.threadid()])
    end
    tseed!(rngs, 1)
    Threads.@threads for i = 1:length(x)
        y[i] = rand(rngs[Threads.threadid()])
    end
    @test x == y
    states = map(rng -> rng.state, rngs)
    @test length(unique(states)) == Threads.nthreads()
end

@testset "nblasthreads/@with_blasthreads" begin
    BLAS.set_num_threads(4)
    @test nblasthreads() == 4
    BLAS.set_num_threads(1)
    @test nblasthreads() == 1
    @with_blasthreads 6 begin
        @test nblasthreads() == 6
    end
    @test nblasthreads() == 1
end

end
