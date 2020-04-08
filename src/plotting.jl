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

function termplot(x::AbstractVector, y::AbstractVector, name = nothing, args...; kwargs...)
    termplot(Line(x, y, name), args...; kwargs...)
end

function termplot(y::AbstractVector, name = nothing, args...; kwargs...)
    termplot(Line(y, name), args...; kwargs...)
end

function termplot(
    lines::Line...;
    title = "",
    width = 60,
    height = 10,
    normalize = false,
    maxpoints = typemax(Int),
)
    isbad = any(lines) do l
        any(v -> isnan(v) || isinf(v), l.x)
        any(v -> isnan(v) || isinf(v), l.y)
    end
    isbad && throw(ArgumentError("NaN or Inf detected"))


    xmax = maximum(l -> maximum(l.x), lines)
    winmin = xmax - maxpoints
    lines = map(lines) do l
        minidx = findfirst(i -> i > winmin, l.x)
        minidx = isnothing(minidx) ? 1 : minidx
        Line(l.x[minidx:end], l.y[minidx:end], l.name)
    end
    xmin = minimum(l -> minimum(l.x), lines)


    # shift/scale lines to fall into [-1, 1]
    if normalize
        names = map(lines) do l
            minn, maxx = extrema(l.y)
            @sprintf "%s=%.3g [%.3g, %.3g]" l.name l.y[end] minn maxx
        end
        lines = map(lines) do l
            Line(l.x, scaleandcenter!(Float64.(l.y), center = 0, range = 2), l.name)
        end
    else
        names = map(l -> l.name, lines)
    end

    ymin = round(minimum(l -> minimum(l.y), lines), digits = 2)
    ymax = round(maximum(l -> maximum(l.y), lines), digits = 2)

    l = lines[1]
    plt = UnicodePlots.lineplot(
        l.x,
        l.y;
        name = names[1],
        xlim = [xmin, xmax],
        ylim = [ymin, ymax],
        title = title,
        width = width,
        height = height,
    )
    for i = 2:length(lines)
        l = lines[i]
        UnicodePlots.lineplot!(plt, l.x, l.y, name = names[i])
    end
    plt
end
