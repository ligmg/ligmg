using MAT
using MatrixMarket

function countz(i)
    local count = 0
    for e in i
        if e != 0
            count += 1
        end
    end
    count
end

function write_graph(A, filename)
    open(filename, "w") do out
        x,y,z=findnz(A)

        A = sparse(x,y,map(a->round(Int, abs(a)), z))

        count = 0

        rows = rowvals(A)
        vals = nonzeros(A)
        x,y = size(A)
        d = countz(diag(A))
        write(out, "$x $(round(Int, (nnz(A)-d)/2))\n")
        for i in 1:y
            for j in nzrange(A, i)
                if rows[j] != i
                    count += 1
                    write(out, " $(rows[j])")
                end
            end
            write(out, "\n")
        end
    end
end

file = ARGS[1]
ext = splitext(file)[2]
A = if ext == ".mtx"
        mmread(file)
    else
        P = matread(file)["Problem"]
        A = P["A"]
    end
if all(x -> x == 0, sum(A, 1))
    A = spdiagm(diag(A)) - A
end
write_graph(A, ARGS[2])
