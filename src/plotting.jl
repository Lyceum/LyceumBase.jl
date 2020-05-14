function termplot(
    xs::AbstractVector{<:AbstractVector},
    ys::AbstractVector{<:AbstractVector};
    title::AbstractString = "",
    width::Integer = 100,
    height::Integer = 15,
    normalize::Bool = false,
    labels = nothing,
)

    for (x, y) in zip(xs, ys)
        all(isfinite, x) && all(isfinite, y) || throw(DomainError("NaN or Inf detected"))
        size(x) == size(y) || throw(DimensionMismatch("x and y must have the same size"))
    end

    _labels = labels === nothing ? Tuple("y$i" for i = 1:length(xs)) : labels
    length(_labels) == length(xs) || argerror("length of labels does not match number of lines")

    lines = zip(xs, ys, _labels)

    # shift/scale lines to fall into [-1, 1]
    if normalize
        lines = map(lines) do (x, y, name)
            _min, _max = extrema(y)
            name = @sprintf "%s [%.3g, %.3g]" name _min _max
            y = scaleandcenter!(convert(AbstractVector{Float64}, copy(y)), center = 0, range = 2)
            x, y, name
        end
    end

    xmin = round(minimum(l -> minimum(l[1]), lines), digits = 2)
    xmax = round(maximum(l -> maximum(l[1]), lines), digits = 2)
    ymin = round(minimum(l -> minimum(l[2]), lines), digits = 2)
    ymax = round(maximum(l -> maximum(l[2]), lines), digits = 2)

    # account for legend in plot width
    legend_len = maximum(l -> length(l[3]), lines)
    width -= legend_len

    (x, y, name), rest = firstrest(lines)
    plt = UnicodePlots.lineplot(
        x,
        y,
        name = name,
        xlim = [xmin, xmax],
        ylim = [ymin, ymax],
        title = title,
        width = width,
        height = height,
    )
    for (x, y, name) in rest
        UnicodePlots.lineplot!(plt, x, y, name = name)
    end

    return plt
end
