struct Experiment
    file::JLSOFile
    savepath::String
    overwrite::Bool
end

#function setindex!(jlso::JLSOFile, value, name::Symbol)

function Experiment(
    savepath::AbstractString;
    overwrite::Bool = false,
    paths::AbstractVector{<:AbstractString} = String[],
    repos::AbstractVector{<:AbstractString} = String[],
    copy_dirty = false,
    format::Symbol = :julia_serialize,
    compression::Symbol = :none,
    strict = false
)
    isdir(savepath) && throw(ArgumentError("$savepath is a directory"))

    ext = last(splitext(savepath))
    if ext == ""
        @warn "Extension not specified. Adding \".jlso\" extension."
        savepath *= ".jlso"
    elseif !(ext == ".jlso")
        throw(ArgumentError("Only .jlso extension supported"))
    end

    file = JLSOFile(Dict{Symbol,Any}(), format = format, compression = compression)

    savepath = abspath(normpath((expanduser(savepath))))
    if isfile(savepath)
        if overwrite
            @info "Deleting $savepath"
            rm(savepath)
        else
            newpath = mkgoodpath(savepath)
            @warn "$savepath already exists and overwrite is false. Using $newpath instead"
            savepath = newpath
        end
    end

    meta = Dict{Symbol, Any}()
    meta[:starttime] = string(now())
    meta[:versioninfo] = versionstring()

    repos, notrepos = _processrepos(repos, copy_dirty, strict)
    meta[:repos] = repos
    meta[:paths] = _processpaths(vcat(paths, notrepos), strict)

    file[:meta] = meta

    Experiment(file, savepath, overwrite)
end


Base.getindex(exp::Experiment, name::Symbol) = getindex(exp.file, _handlecollision(exp.file, name))
function Base.setindex!(exp::Experiment, value, name::Symbol)
    setindex(exp.file, value, _handlecollision(exp.file, name))
end


function finish!(exp::Experiment)
    savepath = exp.savepath
    if isfile(savepath)
        if exp.overwrite
            @info "$savepath exists. Deleting."
            rm(savepath, force=true)
        else
            newpath = mkgoodpath(savepath)
            @warn "$savepath exists, using $newpath instead"
            savepath = newpath
        end
    end
    open(savepath, "w") do io
        write(io, exp.file)
    end
    @info "Experiment saved to $savepath"
    exp
end

function _processrepos(paths, copy_dirty, strict)
    repos = Dict{String,Any}()
    notrepos = String[]
    for path in paths
        path = String(path)
        if haskey(repos, path)
            msg = "Duplicate repo $path detected"
            strict ? error(msg) : @warn msg
        elseif isrepo(path)
            repos[path] = repometa(path, copy_dirty=copy_dirty)
        elseif isfile(path) || isdir(path)
            msg = "expected a git repo but got $path. Copying instead."
            strict ? error(msg) : @warn msg
            push!(notrepos, path)
        else
            msg = "Skipping $path: does not exist"
            strict ? error(msg) : @warn msg
        end
    end
    repos, notrepos
end

function _processpaths(paths, strict)
    Pkg.PlatformEngines.probe_platform_engines!()
    d = Dict{String, Any}()
    for path in paths
        path = String(path)
        if haskey(d, path)
            msg = "Duplicate path $path detected"
            strict ? error(msg) : @warn msg
        elseif isfile(path)
            d[path] = read(path)
        elseif isdir(path)
            tmp = tempname()
            try
                run(Pkg.PlatformEngines.gen_package_cmd(path, tmp))
                d[path] = (type="tar.gz", data=read(tmp))
            catch e
                bt = catch_backtrace()
                if strict
                    rethrow(e)
                else
                    @error sprint(io -> showerror(io, e, bt))
                end
            finally
                rm(tmp, force=true)
            end
        else
            msg = "Skipping $path: does not exist"
            strict ? error(msg) : @warn msg
        end
    end
    d
end


function _handlecollision(x::JLSOFile, k::Symbol)
    if haskey(x.objects, k)
        i = 1
        newk = Symbol(k, :_collision, i)
        while haskey(x.objects, newk)
            i += 1
            newk = Symbol(k, :_collision, i)
        end
        @warn "$k already found in this Experiment. Renaming entry to $newk."
        k = newk
    end
    k
end