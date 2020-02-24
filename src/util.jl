macro mustimplement(sig)
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end

@inline propertytype(obj, name::Symbol) = typeof(getproperty(obj, name))