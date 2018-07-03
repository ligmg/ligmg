# Convert a prolongation matrix P to a tentative operator S.
# P is piecewise constant over some number of aggregates.
# S has a n x n for each aggregate where n is the size of the aggregate.

using PETScBinaryIO

P = readPETSc(ARGS[1])

rows = rowvals(P)
m, n = size(P)
is = Vector{Int}()
js = Vector{Int}()
ks = Vector{Float64}()
j = 0
for i = 1:n
    nzrng = nzrange(P,i)
    agg_size = length(nzrng)
    if agg_size > 10000
        println("Aggregate is too large ($agg_size)")
        exit(-1)
    end
    block = fill(-1/agg_size, (agg_size, agg_size)) # B(BᵀB)⁻¹Bᵀ
    block[diagind(block)] += 1 # I - B(BᵀB)⁻¹Bᵀ

    U, _, _ = svd(block)
    for x = 1:agg_size
        for y = 1:agg_size-1
            push!(is, rows[nzrng[x]])
            push!(js, j + y)
            push!(ks, U[x,y])
        end
    end
    j += agg_size-1
end
S = sparse(is,js,ks,m,j)

writePETSc(ARGS[2], S)
