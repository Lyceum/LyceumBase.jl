"""
    delta(x)

Compute the pair-wise difference between adject elements in `x`,
where delta(x)[i] = x[i+1]  - x[i]
"""
delta(x) = x[2:end] .- x[1:end-1]

function scaleandcenter!(x::AbstractArray; center = 0, range = 1)
    minn, maxx = extrema(x)
    if maxx ≈ minn
        x .= center
    else
        @. x = ((x - minn) / (maxx - minn) - 1 // 2) * range + center
    end
    x
end

zerofn(x) = 0
zerofn(x1, x2) = 0
zerofn(x1, x2, x3) = 0
zerofn(x1, x2, x3, xs...) = 0

noop(x) = nothing
noop(x1, x2) = nothing
noop(x1, x2, x3) = nothing
noop(x1, x2, x3, xs...) = nothing

@inline function symmul!(
    C::Symmetric{T},
    transA::Transpose{<:Any,<:StridedVecOrMat{T}},
    B::StridedVecOrMat{T},
    alpha::Number = true,
    beta::Number = false,
) where {T<:LinearAlgebra.BlasFloat}
    A = transA.parent
    if A === B
        #alpha, beta = promote(alpha, beta, zero(T))
        alpha = convert(T, alpha)
        beta = convert(T, beta)
        BLAS.syrk!(C.uplo, 'T', alpha, A, beta, C.data)
        return C
    else
        return LinearAlgebra.gemm_wrapper!(C, 'T', 'N', A, B, alpha, beta)
    end
end

@inline function symmul!(
    C::Symmetric{T},
    A::StridedVecOrMat{T},
    transB::Transpose{<:Any,<:StridedVecOrMat{T}},
    alpha::Number = true,
    beta::Number = false,
) where {T<:LinearAlgebra.BlasFloat}
    B = transB.parent
    if A === B
        #alpha, beta = promote(alpha, beta, zero(T))
        alpha = convert(T, alpha)
        beta = convert(T, beta)
        BLAS.syrk!(C.uplo, 'N', alpha, B, beta, C.data)
        return C
    else
        return LinearAlgebra.gemm_wrapper!(C, 'N', 'T', A, B, alpha, beta)
    end
end


macro forwardfield(ex, fs)
    @capture(ex, T_.field_) || error("Syntax: @forward T.x f, g, h")
    T = esc(T)
    fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
    :(
        $(
            [
                :(
                    $f(x::$T, args...; kwargs...) = (Base.@_inline_meta;
                    $f(getfield(x, $(QuoteNode(field))), args...; kwargs...))
                )
                for f in fs
            ]...
        );
        nothing
    )
end

isnaninf(x::Number) = isinf(x) || isnan(x)

function namedtuplify(x)
    names = fieldnames(typeof(x))
    NamedTuple{names}(getfield(x, n) for n in names)
end

"""
    wraptopi(theta::Real)

Wrap the angle `theta`, specified in radians, to the interval [-pi, pi)
"""
@inline wraptopi(theta::Real) = mod1(theta + pi, 2pi) - pi


tuplecat(t1, t2, t3...) = tuplecat((t1..., t2...), t3...)
function tuplecat(t1::NamedTuple, t2::NamedTuple, t3::NamedTuple...)
    tuplecat(NamedTuple{tuple(keys(t1)..., keys(t2)...)}(tuple(t1..., t2...)), t3...)
end
tuplecat(t) = t


mutable struct Converged{T}
    "absolute tolerance"
    tol::T
    lastval::T
    initialized::Bool
    Converged{T}(tol) where {T<:Number} = new{T}(T(tol), zero(T), false)
end
Converged(tol) = Converged{typeof(tol)}(tol)

function (x::Converged)(val::Real)
    ret = x.initialized && abs(val - x.lastval) < x.tol
    x.initialized = true
    x.lastval = val
    ret
end

LyceumBase.reset!(x::Converged{T}) where {T} =
    (x.initialized = false; x.lastval = zero(T); x)


macro noalloc(expr)
    quote
        local tmp = @benchmark $expr samples = 1 evals = 1
        @test iszero(tmp.allocs)
    end
end

function mkgoodpath(filepath::String; force::Bool = false, sep = '_')
    isdir(filepath) && throw(ArgumentError("$filepath is a directory"))

    if isfile(filepath)
        if force
            rm(filepath)
            return filepath
        else
            val = 1
            dir, file = splitdir(filepath)
            file, ext = splitext(file)
            f() = joinpath(dir, "$(file)$(sep)$(val)$(ext)")
            newpath = f()
            while isfile(newpath)
                val += 1
                newpath = f()
            end
            return newpath
        end
    else
        return filepath
    end
end


function filter_nt(f, nt::NamedTuple)
    ks = Symbol[]
    for (k, v) in pairs(nt)
        f(k, v) && push!(ks, k)
    end
    NamedTuple{Tuple(ks)}(Tuple(getfield(nt, k) for k in ks))
end

function filter_nt(
    nt::NamedTuple;
    include::Union{Symbol,TupleN{Symbol}} = (),
    exclude::Union{Symbol,TupleN{Symbol}},
)
    include isa Symbol && (include = (include,))
    exclude isa Symbol && (exclude = (exclude,))
    ks = Symbol[]
    for (k, v) in pairs(nt)
        inc = k in include
        exc = k in exclude
        inc && exc && throw(ArgumentError("Symbol $k found in both include and exclude"))
        inc || !exc && push!(ks, k)
    end
    NamedTuple{Tuple(ks)}(Tuple(getfield(nt, k) for k in ks))
end


struct KahanSum{T<:Number}
    sum::T
    c::T
    n::Int
end
KahanSum(T::Type{<:Number} = Float64) = KahanSum(T(0), T(0), 0)
KahanSum(sum::T) where {T<:Number} = KahanSum(sum, T(0), 1)

Statistics.mean(s::KahanSum) = s.sum / s.n
Base.sum(s::KahanSum) = s.sum + s.c

Base.:(+)(x::KahanSum, y::Number) = _add(x, y)
Base.:(+)(x::Number, y::KahanSum) = _add(y, x)
Base.:(+)(x::KahanSum, y::KahanSum) = _add(x, y)

function _add(s::KahanSum{T}, x::U) where {T,U<:Number}
    V = Base.promote_op(+, T, U)
    sum, c, n = convert(V, s.sum), convert(V, s.c), s.n
    x = convert(V, x)

    t = sum + x
    c += ifelse(abs(sum) >= abs(x), (sum - t) + x, (x - t) + sum)

    KahanSum{V}(t, c, n + 1)
end

function _add(s1::KahanSum{T}, s2::KahanSum{U}) where {T,U}
    V = Base.promote_op(+, T, U)

    sum1, c1, n1 = convert(V, s1.sum), convert(V, s1.c), s1.n
    sum2, c2, n2 = convert(V, s2.sum), convert(V, s2.c), s2.n

    sum = _addkbn(sum1, sum2)
    c = _addkbn(c1, c2)
    KahanSum{V}(sum, c, n1 + n2)
end

function _addkbn(x, y)
    t = x + y
    c = ifelse(abs(x) >= abs(y), (x - t) + y, (y - t) + x)
    t - c
end


perturb!(A::AbstractArray) = perturb!(Random.default_rng(), A)
perturbn!(A::AbstractArray) = perturbn!(Random.default_rng(), A)
perturb!(s::Sampleable, A::AbstractArray) = perturb!(Random.default_rng(), s, A)

perturb(rng::AbstractRNG, A::AbstractArray) = perturb!(rng, copy(A))
perturbn(rng::AbstractRNG, A::AbstractArray) = perturbn!(rng, copy(A))
perturb(rng::AbstractRNG, s::Sampleable, A::AbstractArray) = perturb!(rng, s, copy(A))

function perturb!(rng::AbstractRNG, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += rand(rng)
    end
    A
end

function perturbn!(rng::AbstractRNG, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += randn(rng)
    end
    A
end

function perturb!(rng::AbstractRNG, s::Sampleable, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += rand(rng, s)
    end
    A
end
