function blocks(A, rows)
    block_size = div(size(A,1),rows)
    blks = Array{SparseMatrixCSC, 2}(rows, rows)
    for x in 1:rows
        for y in 1:rows
            xr = (x-1)*block_size+1 : ((x == rows) ? size(A,1) : x * block_size)
            yr = (y-1)*block_size+1 : ((y == rows) ? size(A,1) : y * block_size)
            blks[x,y] = A[xr,yr]
        end
    end

    blks
end
