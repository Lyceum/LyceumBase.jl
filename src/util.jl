macro mustimplement(sig)
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end