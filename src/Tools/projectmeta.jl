function environment(; repospec = nothing, verbose_versioninfo = false)
    project, manifest, verinfo = projectmeta(verbose = verbose_versioninfo)
    meta = Dict{Symbol,Any}(
        :project => project,
        :manifest => manifest,
        :versioninfo => verinfo,
    )
    if !isnothing(repospec)
        meta[:repoinfo] = repometa(repospec)
    end
    meta
end

function projectmeta(; verbose = false)
    project = read(Base.active_project(), String)
    manifest = read(joinpath(dirname(Base.active_project()), "Manifest.toml"), String)
    verinfo = versionstring()
    project, manifest, verinfo
end

function versionstring(verbose = false)
    io = IOBuffer()
    versioninfo(io, verbose = verbose)
    String(take!(io))
end

function isrepo(path::AbstractString)
    ispath(path) || throw(ArgumentError("$path does not exist"))
    try
        GitRepo(path)
        return true
    catch
        return false
    end
end

function repometa(path::AbstractString; copy_dirty = false)
    repo = LibGit2.GitRepo(path)
    meta = Dict{Symbol,Any}()
    meta[:remotes] = remoteinfo(repo)
    meta[:HEAD] = headhash(repo)
    meta[:index_dirty] = LibGit2.isdirty(repo, cached = true)
    meta[:workingtree_dirty] = LibGit2.isdirty(repo, cached = false)

    if copy_dirty
        dirty = Dict{String,Vector{UInt8}}()
        for (relpath, fullpath) in walkdirty(repo, untracked = true)
            dirty[relpath] = read(fullpath)
        end
        meta[:dirtyfiles] = dirty
    end
    meta
end

function remoteinfo(repo::GitRepo)
    names = LibGit2.remotes(repo)
    info = Dict{String,String}()
    for name in names
        remote = LibGit2.get(LibGit2.GitRemote, repo, name)
        info[name] = LibGit2.url(remote)
    end
    info
end

function headhash(repo::GitRepo)
    commit = LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))
    string(LibGit2.GitHash(commit))
end

function isuntracked(repo::GitRepo, path)
    status = LibGit2.status(repo, path)
    status !== nothing && status &
                          LibGit2.Consts.STATUS_WT_NEW == LibGit2.Consts.STATUS_WT_NEW
end

function walkdirty(repo::GitRepo; untracked::Bool = false)
    repodir = LibGit2.workdir(repo)
    gitdir = LibGit2.gitdir(repo)

    ignore = String[]
    for (root, _, files) in walkdir(gitdir), file in files
        push!(ignore, realpath(joinpath(root, file)))
    end

    dirty = Vector{Tuple{String,String}}()
    for (root, _, files) in walkdir(repodir), file in files
        fullpath = realpath(joinpath(root, file))
        if !(fullpath in ignore)
            pathspec = relpath(fullpath, repodir)
            if LibGit2.isdirty(repo, pathspec; cached = true) || LibGit2.isdirty(
                repo,
                pathspec;
                cached = false,
            ) || (untracked && isuntracked(repo, pathspec))
                push!(dirty, (relpath(fullpath, repodir), fullpath))
            end
        end
    end
    dirty
end
