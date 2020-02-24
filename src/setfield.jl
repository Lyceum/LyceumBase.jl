# TODO relies on implementation details of BangBang for pure versions (_setindex)
module SetfieldImpl

using BangBang: NoBang, BangBang, implements
using Setfield: ComposedLens,
                ConstIndexLens,
                DynamicIndexLens,
                IndexLens,
                Lens,
                PropertyLens,
                Setfield,
                set
using ..LyceumBase: propertytype
using ..DocStringExtensions


export @set!, prefermutation


const SupportedIndexLens = Union{ConstIndexLens,DynamicIndexLens,IndexLens}


struct Lens!{T<:Lens} <: Lens
    lens::T
end


Setfield.get(obj, lens::Lens!) = get(obj, lens.lens)

function Setfield.set(obj, lens::Lens!, value)
    if issettable(obj, lens.lens)
        _set(obj, lens, value)
    else
        error("At least one of the applied lens must correspond to a mutable object")
    end
end


function _set(obj, lens::Lens!{<:ComposedLens}, value)
    inner_lens, outer_lens = Lens!(lens.lens.inner), Lens!(lens.lens.outer)
    inner_obj = get(obj, outer_lens)
    inner_val = _set(inner_obj, inner_lens, value)
    _set(obj, outer_lens, inner_val)
end

function _set(obj, ::Lens!{<:PropertyLens{name}}, value) where {name}
    if implements(setproperty!, obj)
        setproperty!(obj, name, value)
        return obj
    else
        return NoBang.setproperty(obj, name, value)
    end
end

indicesfor(lens::IndexLens, _) = lens.indices
indicesfor(::ConstIndexLens{I}, _) where {I} = I
indicesfor(lens::DynamicIndexLens, obj) = lens.f(obj)

function _set(obj, lens::Lens!{<:SupportedIndexLens}, value)
    if implements(setindex!, obj)
        setindex!(obj, value, indicesfor(lens.lens, obj)...)
        return obj
    else
        return NoBang._setindex(obj, value, indicesfor(lens.lens, obj)...)
    end
end


"""
    $(TYPEDSIGNATURES)

See also [`@set!`](@ref).
"""
prefermutation(lens::Lens) = Lens!(lens)

"""
    $(TYPEDSIGNATURES)

Like `Setfield.@set`, but always mutate when at least one of the nested objects is
mutable. Otherwise fails.

# Examples
```jldoctest
julia> using LyceumBase

julia> mutable struct Mutable
           a
           b
       end

julia> x = orig = Mutable((x=Mutable(1, 2), y=3), 4);

julia> @set! x.a.x.a = 10;

julia> @assert x.a.x.a == orig.a.x.a == 10

julia> immutable = (x=(y=(z=1,),),)

julia> # this would error: @set! y.x.a = 10;
```
"""
macro set!(ex)
    Setfield.setmacro(prefermutation, ex, overwrite = false)
end


Base.@propagate_inbounds issettable(obj, ::PropertyLens) = implements(setproperty!, obj)
Base.@propagate_inbounds issettable(obj, ::SupportedIndexLens) = implements(setindex!, obj)
Base.@propagate_inbounds function issettable(obj, lens::ComposedLens)
    issettable(obj, lens.outer) || issettable(get(obj, lens.outer), lens.inner)
end

end
