using PyCall
@pyimport scipy.linalg.interpolative
println("Julia loaded")

function main(rows, cols, values)
    println("In julia")
    A = sparse(rows, cols, values)
    println(interpolative.interp_decomp(A, 2))
    println("Exiting julia")
end
