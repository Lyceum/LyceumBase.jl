const TJUMP = big(10)^20

function seed_threadrngs!(seed = Random.make_seed(); jump = TJUMP)
    seed_threadrngs!(Random.THREAD_RNGs, seed, jump = jump)
end

function seed_threadrngs!(
    rngs::Vector{MersenneTwister},
    seed = Random.make_seed();
    jump = TJUMP,
)
    Threads.threadid() == 1 || throw(ArgumentError("Can only call `seed_threadrngs!` from main thread"))
    if length(rngs) != Threads.nthreads()
        throw(ArgumentError("length(rngs) must equal Threads.nthreads()"))
    end

    rngs[1] = MersenneTwister(seed)
    Threads.@threads for _ = 1:Threads.nthreads()
        tid = Threads.threadid()
        if tid > 1
            rngs[tid] = randjump(rngs[1], jump * (tid - 1))
        end
    end
    rngs
end

function threadrngs(seed = Random.make_seed(); jump = TJUMP)
    trngs = Vector{MersenneTwister}(undef, Threads.nthreads())
    seed_threadrngs!(trngs, seed, jump = jump)
end


getrange(N::Int, np::Int, i::Int = Threads.threadid()) = getrange(1:N, np, i)
function getrange(range::UnitRange{Int}, np::Int, i::Int = Threads.threadid())
    N = length(range)
    start = first(range)
    each, extras = divrem(N, np)
    if each == 0
        from = start + i - 1
        return ifelse(i <= extras, from:from, nothing)
    else
        from = (i - 1) * each + min(extras, i - 1) + start
        to = from + each - 1 + ifelse(i ≤ extras, 1, 0)
        return from:to
    end
end

splitrange(N::Int, np::Int) = splitrange(1:N, np)
function splitrange(range::UnitRange{Int}, np::Int)
    d, r = divrem(length(range), np)
    np = d == 0 ? r : np
    ntuple(i -> getrange(range, np, i), np)
end


function nblasthreads()
    ccall(
        (BLAS.@blasfunc(openblas_get_num_threads), Base.libblas_name),
        LinearAlgebra.BlasInt,
        (),
    )
end

macro with_blasthreads(nthreads, expr)
    quote
        local n = nblasthreads()
        BLAS.set_num_threads($(esc(nthreads)))
        $(esc(expr))
        BLAS.set_num_threads(n)
    end
end
