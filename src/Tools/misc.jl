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

LyceumBase.reset!(x::Converged{T}) where {T} = (x.initialized = false; x.lastval = zero(T); x)


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


perturb!(A::AbstractArray) = perturb!(Random.default_rng(), A)
perturbn!(A::AbstractArray) = perturb!(Random.default_rng(), A)
perturb!(s::Sampleable, A::AbstractArray) = perturb!(Random.default_rng(), s, A)

perturb(rng::AbstractRNG, A::AbstractArray) = perturb!(rng, copy(A))
perturbn(rng::AbstractRNG, A::AbstractArray) = perturb!(rng, copy(A))
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
