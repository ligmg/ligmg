using HDF5

d = h5read(ARGS[1], "/")
x = d["eigen_vector_0"]
for i in 1:(length(d["eigen_values"])-1)
    x += d["eigen_vector_$i"]
end

println(length(x))
for i in x
    println(i)
end
