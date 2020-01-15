@testset "splitrange/getrange" begin
    let N = 16, np = 4
        x = splitrange(N, np)
        @test x == (1:4, 5:8, 9:12, 13:16)
        for i = 1:np
            @test x[i] == getrange(N, np, i)
        end
    end

    let N = 4, np = 16
        x = splitrange(N, np)
        @test x == (1:1, 2:2, 3:3, 4:4)
        for i = 1:N
            @test x[i] == getrange(N, np, i)
        end
    end
end

@testset "seed_threadrngs!" begin
    # Test that naievely calling Random.seed! in a single thread doesn't
    # seed all threads
    x = zeros(1000)
    y = zeros(1000)
    Random.seed!(1)
    Threads.@threads for i=1:length(x)
       y[i] = rand()
    end
    Random.seed!(1)
    Threads.@threads for i=1:length(x)
       x[i] = rand()
    end
    @test x != y

    x = zeros(1000)
    y = zeros(1000)
    seed_threadrngs!(1)
    Threads.@threads for i=1:length(x)
       y[i] = rand()
    end
    seed_threadrngs!(1)
    Threads.@threads for i=1:length(x)
       x[i] = rand()
    end
    @test x == y

    states = map(rng -> rng.state, Random.THREAD_RNGs)
    @test length(unique(states)) == Threads.nthreads()
end

@testset "nblasthreads/with_blasthreads" begin
    nt = Threads.nthreads()
    if nt > 1
        nb = min(16, nt) # Julia's OpenBLAS allows a max of 16 threads
        BLAS.set_num_threads(nb)
        @test nblasthreads() == nb
        BLAS.set_num_threads(1)
        @test nblasthreads() == 1
        @with_blasthreads nb begin
            @test nblasthreads() == nb
        end
        @test nblasthreads() == 1
    end
end
