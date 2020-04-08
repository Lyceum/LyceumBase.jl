const TJUMP = big(10)^20

"""
    tseed!([rngs=Random.THREAD_RNGs], seed=Random.make_seed(); jump=big(10)^20)

Produce approximately statistically independent rngs by seeding `rngs[1]` with `seed` and
advancing each the state of the remaining rngs by a multiple of `jump`.
"""
@inline function tseed!(seed = Random.make_seed(); jump = TJUMP)
    tseed!(Random.THREAD_RNGs, seed, jump = jump)
end

@noinline function tseed!(
    rngs::Vector{MersenneTwister},
    seed::Union{Integer,Vector{UInt32}} = Random.make_seed();
    jump::Integer = TJUMP,
)
    nt = Threads.nthreads()
    Threads.threadid() == 1 || argerror("Can only call `tseed!` from main thread")
    length(rngs) == nt || argerror("length(rngs) != Threads.nthreads()")

    rngs[1] = MT = MersenneTwister(seed)
    Threads.@threads for _ = 1:nt
        tid = Threads.threadid()
        tid > 1 && (rngs[tid] = randjump(MT, jump * (tid - 1)))
    end
    return rngs
end


getrange(N::Int, np::Int, i::Int = Threads.threadid()) = splitrange(1:N, np, i)
function getrange(range::UnitRange{Int}, np::Int, i::Int = Threads.threadid())
    np > 0 || argerror("np must be > 0")
    0 < i <= np || argerror("i must be in range (0, np]")

    N = length(range)
    start = first(range)
    each, extras = divrem(N, np)
    if each == 0
        from = start + i - 1
        return from:from
    else
        from = (i - 1) * each + min(extras, i - 1) + start
        to = from + each - 1 + ifelse(i <= extras, 1, 0)
        return from:to
    end
end

splitrange(N::Int, np::Int) = splitrange(1:N, np)
function splitrange(range::UnitRange{Int}, np::Int)
    d, r = divrem(length(range), np)
    np = d == 0 ? r : np
    UnitRange{Int}[getrange(range, np, i) for i = 1:np]
end

"""
    $(SIGNATURES)

Get the current number of threads currently used by the BLAS library.
"""
@inline function nblasthreads()
    ccall((BLAS.@blasfunc(openblas_get_num_threads), Base.libblas_name), LinearAlgebra.BlasInt, ())
end

"""
    @with_blasthreads n expression

Temporarily set the number of threads the BLAS library should use to `n` within the `expression`.

See also: [`nblasthreads`](@ref).

# Examples
```jldoctest
julia> nblasthreads()
6

julia> @with_blasthreads 10 println("Using ", nblasthreads(), " BLAS threads")
Using 10 BLAS threads

julia> nblasthreads()
6
```
"""
macro with_blasthreads(nthreads, expr)
    quote
        local n = nblasthreads()
        BLAS.set_num_threads($(esc(nthreads)))
        $(esc(expr))
        BLAS.set_num_threads(n)
    end
end
