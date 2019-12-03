function _foreachfield(r::Base.OneTo{Int})
    exprs = Expr[]
    for i in r
        get_ith = Expr(:call, :getfield, :xs, i)
        f = Expr(:call, :f, get_ith)
        push!(exprs, f)
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

function _foreachfield(r1::Base.OneTo{Int}, r2::Base.OneTo{Int})
    exprs = Expr[]
    for i in r2
        f = Expr(:call, :f)
        for j in r1
            get_ith = Expr(:call, :getfield, :xs, j)
            get_jth = Expr(:call, :getfield, get_ith, i)
            push!(f.args, get_jth)
        end
        push!(exprs, f)
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

@generated foreachfield(f, xs::NTuple{N,Any}) where {N} = _foreachfield(Base.OneTo(N))
@generated foreachfield(f, xs::NamedTuple{names}) where {names} =
    _foreachfield(Base.OneTo(length(names)))

@generated foreachfield_tabular(f, xs::NTuple{M,NTuple{N,Any}}) where {M,N} =
    _foreachfield(Base.OneTo(M), Base.OneTo(N))
@generated foreachfield_tabular(f, xs::NamedTuple{names,<:NTuple{N,Any}}) where {names,N} =
    _foreachfield(Base.OneTo(length(names)), Base.OneTo(N))
@generated foreachfield_tabular(f, xs::NTuple{M,NamedTuple{names}}) where {M,names} =
    _foreachfield(Base.OneTo(M), Base.OneTo(length(names)))
@generated foreachfield_tabular(
    f,
    xs::NamedTuple{n1,T},
) where {n1,n2,T<:Tuple{Vararg{NamedTuple{n2}}}} =
    _foreachfield(Base.OneTo(length(n1)), Base.OneTo(length(n2)))

function _mapfield(r::Base.OneTo{Int})
    exprs = Expr[]
    for i in r
        ex1 = Expr(:call, :getfield, :xs, i)
        ex2 = Expr(:call, :f, ex1)
        push!(exprs, ex2)
    end
    return Expr(:tuple, exprs...)
end

@generated mapfield(f, xs::NTuple{N,Any}) where {N} = _mapfield(Base.OneTo(N))
@generated function mapfield(f, xs::NamedTuple{names}) where {names}
    ex = _mapfield(Base.OneTo(length(names)))
    :(NamedTuple{$names}($ex))
end
