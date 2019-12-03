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

Threads.@threads for _ = 1:Threads.nthreads()
    Random.seed!(1)
end
let states = map(rng -> rng.state, Random.THREAD_RNGs)
    @test length(unique(states)) == 1
end
seed_threadrngs!(1)
let states = map(rng -> rng.state, Random.THREAD_RNGs)
    @test length(unique(states)) == Threads.nthreads()
end

let rngs = threadrngs()
    states = map(rng -> rng.state, rngs)
    @test length(unique(states)) == Threads.nthreads()
end

let nt = Threads.nthreads()
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
