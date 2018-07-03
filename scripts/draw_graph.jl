# Draw a series of graphs as svg files

using LightGraphs
using GraphPlot
using PETScBinaryIO
using Compose
using Colors
using HDF5
using MatrixMarket
using ArgParse
using PerceptualColourMaps

s = ArgParseSettings()
@add_arg_table s begin
    "--eig"
        action = :store_true
        help = "color nodes by eigenvalue"
    "INPUT.petsc"
        help = "input graph"
        required = true
    "OUTPUT.pdf"
        help = "output filename"
        required = true
end
parsed_args = parse_args(ARGS, s)

function write_chaco(filename, g)
    open(filename, "w") do f
        write(f, "$(nv(g)) $(ne(g))\n")
        for i = 1:nv(g)
            for n in neighbors(g, i)
                write(f, "$n ")
            end
            write(f, "\n")
        end
    end
end

function map_colormap(vals, clm)
    min = minimum(vals)
    max = maximum(vals)
    map(x -> clm[Int(floor(1+(x - min)/(max-min)*(length(clm)-1)))], vals)
end

function compute_eig(A)
    mmwrite("tmp.mtx", spdiagm(vec(sum(A,1))) - A)
    run(`build/bin/ligmg_eigensolve tmp.mtx tmp.h5 -eps_smallest_magnitude -st_pc_type ligmg -st_ksp_type preonly -eps_monitor -st_type sinvert -eps_target 0 -eps_nev 1`)
    d = h5read("tmp.h5", "/")
    rm("tmp.h5")
    rm("tmp.mtx")
    d["eigen_vector_0"]
end

function layout_graph(A)
    write_chaco("tmp.graph", A)
    x = readdlm(read(`../ogdf_layout/build/ogdf_layout tmp.graph`))
    rm("tmp.graph")
    x[:,1], x[:,2]
end

println("Reading")
infile = parsed_args["INPUT.petsc"]
A = readPETSc(infile)
A = (A + A.')/2
if all(x->x==0, sum(A,1))
    A = spdiagm(diag(A)) - A
end
g = Graph(A)

plot = if parsed_args["eig"]
    println("Computing smallest eigenvalue")
    eig = compute_eig(A)
    nodefill = map_colormap(eig, cmap("D1"))
    gplot(g; layout=layout_graph, nodefillc=nodefill, NODESIZE=1/sqrt(nv(g)), edgestrokec=RGBA(0,0,0,0.01))
else
    gplot(g; layout=layout_graph, nodefillc=colorant"purple", EDGELINEWIDTH=15/sqrt(nv(g)), NODESIZE=0.5/sqrt(nv(g)), edgestrokec=RGBA(0,0,0,0.3))
end

outfile = parsed_args["OUTPUT.pdf"]
println("Saving to $outfile")
draw(PNG(outfile, 2400px, 2400px, ), plot)
