using Requests
using MatrixDepot
using DataFrames
using DataFramesMeta
using PETScBinaryIO
using MatrixMarket
using MAT

function max_by(fn, itr)
  v = fn(itr[1])
  e = itr[1]
  for i in itr
    vv = fn(i)
    if vv > v
      v = vv
      e = i
    end
  end

  e
end

function largest_component(L)
  cc = connected_components(L)
  lc = sort(max_by(length, cc))
  L[lc,lc]
end

function connected_components(L)
  components = Vector{Vector{Int64}}()
  notfound = Set{Int64}(1:size(L, 1))
  while !isempty(notfound)
    s = first(notfound)
    old = Set{Int64}()
    frontier = Set{Int64}(s)
    new = Set{Int64}(s)
    while !isempty(new)
      union!(old, new)
      frontier = copy(new)
      empty!(new)
      for v in frontier
        for n in findn(L[:, v])
          if !(n in old || n in new || n in frontier)
            push!(new, n)
          end
        end
      end
    end
    push!(components, collect(old))
    setdiff!(notfound, old)
  end

  components
end

"""
Get metadata information about all graphs in the UF sparse matrix collection
"""
function get_ufcollection_metadata()
    csv = String(get("https://www.cise.ufl.edu/research/sparse/matrices/UFstats.csv").data)
    i = findnext(x -> x == '\n', csv, 1)
    i = findnext(x -> x == '\n', csv, i+1)
    df = readtable(IOBuffer(csv[i+1:end]), header=false, names = [:group, :name, :rows, :cols, :nnz, :real, :binary, :geom, :posdef, :psym, :nsym, :kind])
    df[:geom] = df[:geom] .== 1
    df[:posdef] = df[:posdef] .== 1
    df[:binary] = df[:binary] .== 1
    df[:real] = df[:real] .== 1
    df[:id] = 1:nrow(df)
    df
end

graph_kinds = [ "directed weighted graph"
              , "directed graph"
              , "undirected weighted graph"
              , "undirected random graph"
              , "undirected weighted random graph"
              , "duplicate undirected random graph"
              , "directed weighted random graph"
              , "weighted bipartite graph"
              , "undirected graph"
              , "bipartite graph"
              , "directed multigraph"
              , "undirected multigraph"
              , "undirected weighted graph sequence"
              , "undirected bipartite graph"
              , "undirected graph sequence"
              , "directed weighted graph sequence"
              , "weighted undirected graph"
              , "weighted directed graph"
              , "term/document graph"
              , "random undirected graph"
              , "random unweighted graph"
              ]

"""
Filter `df` by `:kind in kinds`
"""
function select_by_kinds(df, kinds)
    mapreduce(k -> @where(df, :kind .== k), vcat, kinds)
end

function download_graphs(filepath, df, formats)
    for (g, n) in zip(df[:group], df[:name])
        if isfile("$filepath/$n.mtx")
            continue
        end
        localpath = "$filepath/$n.tmp.mat"
        println("Downloading $g/$n to $localpath")
        println("https://www.cise.ufl.edu/research/sparse/mat/$g/$n.mat")
        download("https://www.cise.ufl.edu/research/sparse/mat/$g/$n.mat", localpath)
        A = matread(localpath)["Problem"]["A"]
        rm(localpath)
        if size(A, 1) != size(A, 2)
            A = A * A.'
            A = A - spdiagm(diag(A))
        end
        if !issymmetric(A)
            A = (A + A.') ./ 2
        end
        d = sum(A, 2)
        L = spdiagm(vec(d)) - A
        L = largest_component(L)
        for f in formats
            if f == "mat"
                matwrite("$filepath/$n.mat", Dict("L"=>L))
            elseif f == "mtx"
                mmwrite("$filepath/$n.mtx", L)
            elseif f == "petsc"
                writePETSc("$filepath/$n.petsc", L)
            end
        end
    end
end

function download_all_graphs(filepath)
    df = get_ufcollection_metadata()
    gs = select_by_kinds(df, graph_kinds)
    download_graphs(gs, ["mtx"])
end
