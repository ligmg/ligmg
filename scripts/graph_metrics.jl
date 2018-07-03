using LightGraphs
using MatrixMarket
using SimpleWeightedGraphs

name = splitext(splitdir(ARGS[1])[2])[1]
A = mmread(ARGS[1])
A = spdiagm(diag(A)) - A
g = SimpleWeightedGraph(A)

tris = Int(sum(triangles(g))//3)
clustering = global_clustering_coefficient(g)
lc = local_clustering_coefficient(g)
avg_clustering = mean(lc)
#= diam = parallel_diameter(g) =#

println("graph,triangles,clustering,avg_clustering")
@printf "%s,%s,%s,%s" name tris clustering avg_clustering
