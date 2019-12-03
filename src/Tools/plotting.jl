struct Line
    x::AbstractVector
    y::AbstractVector
    name::String
    function Line(x::AbstractVector, y::AbstractVector, name = "")
        x = [x...]
        y = [y...]
        new(x, y, string(name))
    end
end
Line(y::AbstractVector, name = "") = Line(Base.OneTo(length(y)), y, name)

function expplot(x::AbstractVector, y::AbstractVector, name = nothing, args...; kwargs...)
    expplot(Line(x, y, name), args...; kwargs...)
end

function expplot(y::AbstractVector, name = nothing, args...; kwargs...)
    expplot(Line(y, name), args...; kwargs...)
end

function expplot(
    lines::Line...;
    title = "",
    width = 80,
    height = 6,
    normalize = false,
    maxpoints = typemax(Int),
)


    xmax = maximum(l -> maximum(l.x), lines)
    winmin = xmax - maxpoints
    lines = map(lines) do l
        minidx = findfirst(i -> i > winmin, l.x)
        minidx = isnothing(minidx) ? 1 : minidx
        Line(l.x[minidx:end], l.y[minidx:end], l.name)
    end
    xmin = minimum(l -> minimum(l.x), lines)

    extremas = map(l -> round.(extrema(l.y), digits = 2), lines) # range of data before transformations
    curs = map(l -> round(l.y[end], digits = 2), lines)

    # shift/scale lines to fall into [-1, 1]
    if normalize
        lines = map(
            l -> Line(l.x, scaleandcenter!(copy(l.y), center = 0, range = 2), l.name),
            lines,
        )
    end

    ymin = round(minimum(l -> minimum(l.y), lines), digits = 2)
    ymax = round(maximum(l -> maximum(l.y), lines), digits = 2)

    n(l, e, c) = normalize ? "$(l.name)=$c [$(e[1]):$(e[2])]" : l.name

    l = lines[1]
    plt = UnicodePlots.lineplot(
        l.x,
        l.y;
        name = n(l, extremas[1], curs[1]),
        xlim = [xmin, xmax],
        ylim = [ymin, ymax],
        title = title,
        width = width,
        height = height,
    )
    for lidx = 2:length(lines)
        l = lines[lidx]
        e = extremas[lidx]
        c = curs[lidx]
        UnicodePlots.lineplot!(plt, l.x, l.y, name = n(l, e, c))
    end
    plt
end
