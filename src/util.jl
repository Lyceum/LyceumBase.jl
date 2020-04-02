macro mustimplement(sig)
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end

@inline propertytype(x, name::Symbol) = typeof(getproperty(x, name))
