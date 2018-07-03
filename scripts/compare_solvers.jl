# Required packages:
using MATLAB
import IterativeSolvers
using ArgParse
using Iterators

immutable SolveInfo
    x :: Vector
    st_res :: Float64
    residuals :: Vector
    end_res :: Float64
    cyclecomplexity :: Float64
    iters :: Int
    setup_time :: Float64
    solve_time :: Float64
end

"""
Work per digit of accuracy

wda(cycle complexity, change in residual, # of cycles)
"""
wda(cc, dr, cycles) = -1 / log10(ecf(cc, dr, cycles))

"""
Effective convergence factor

ecf(cycle complexity, change in residual, # of cycles)
"""
ecf(cc, dr, cycles) = dr ^ (1/(cc*cycles))

# relaxation is one of sgs, gs, jacobi
# cycleIndex is 1.5 for W, 1 for V
# aggregation is one of affinity-mex, affinity-energy-mex
# minres is true or false
function solve_lamg_matlab(L, b, relaxation, cycle_index, aggregation, minres, tol=1e-8)
    put_variable(:L, L)
    put_variable(:b, b)
    eval_string("""tic;
                   solver = Solvers.newSolver('lamg', 'relaxType', '$relaxation', 'cycleIndex', $cycle_index, 'aggregationUpdate', '$aggregation', 'minRes', $minres);
                   setup = solver.setup('laplacian', L)
                   tsetup = toc;
                   tic;
                   [x, ~, res, details] = solver.solve(setup, b, 'errorReductionTol', $tol);
                   tsolve = toc;
                   """)
    cc = mat"setup.cycleComplexity"
    residuals = mat"details.errorNormHistory"
    iters = mat"length(details.errorNormHistory)"
    tsetup = get_variable(:tsetup)
    tsolve = get_variable(:tsolve)
    x = get_variable(:x)

    SolveInfo(x, norm(b), residuals, norm(b - L*x), cc, iters, tsetup, tsolve)
end

function solve_cmg_matlab(L, b, tol=1e-8)
    put_variable(:L, L)
    put_variable(:b, b)
    eval_string("""tic;
                   solver = Solvers.newSolver('cmg');
                   setup = solver.setup('laplacian', L);
                   tsetup = toc;
                   tic;
                   [x, ~, res, details] = solver.solve(setup, b, 'errorReductionTol', $tol);
                   tsolve = toc;
                   """)
    cc = mat"setup.cycleComplexity"
    residuals = mat"details.errorNormHistory"
    iters = mat"length(details.errorNormHistory)"
    tsetup = get_variable(:tsetup)
    tsolve = get_variable(:tsolve)
    x = get_variable(:x)

    SolveInfo(x, norm(b), residuals, norm(b - L*x), cc, iters, tsetup, tsolve)
end

function solve_jlamg(L, b, tol=1e-8)
    h = build_hierarchy(L)
    x, ci = solve_cg(h, zeros(b), b)
    SolveInfo(x, norm(b), ci.residuals, norm(b - L * x), ci.iteration_complexity, ci.iterations, 0, 0)
end

function solve_augtree(L, b, tol=1e-8)
    f = augTreeLapPrecon(L)
    fact_nnz = nnz(f.subSolver.contents.solvers[1]) # I assume this is the correct way to measure work
    (x, ch) = IterativeSolvers.cg(L, b, Pl=f, tol=tol, maxiter=500, log=true)

    SolveInfo(x, norm(b), ch.data[:resnorm], norm(b - L*x), fact_nnz/nnz(L) + 1, ch.iters)
end

function test_solvers(L, dump_file)
    # construct rhs
    b = rand(size(L,1))
    b = b - mean(b)

    relaxation = ["sgs", "gs", "jacobi"]
    cycle_index = [1.0, 1.5]
    aggregation = [#="affinity",=# "affinity-energy-mex"]
    minres = [true, false]

    #= lamg = solve_lamg_matlab(L, b) =#
    #= cmg = solve_cmg_matlab(L, b) =#
    #= jlamg = solve_jlamg(L, b) =#
    #= augtree = solve_augtree(L, b) =#

    println("    method iters       ∆r    work    wda    setup    solve")
    for (n,f) in [ ("lamg", solve_lamg_matlab)
                 ]
        #
        for (relax, ci, agg, mr) in product(relaxation, cycle_index, aggregation, minres)
            s = f(L, b, relax, ci, agg, mr)
            @printf "%s %f %s %s\n" relax ci agg mr
            @printf "%10s %5d %5.2e %7.2f %6.2f %5.2e %5.2e\n" n s.iters s.end_res/s.st_res s.cyclecomplexity*s.iters wda(s.cyclecomplexity, s.end_res/s.st_res, s.iters) s.setup_time s.solve_time

            if !isa(dump_file, Void)
                s = @sprintf "%s,%d,%e,%f,%f,%e,%e,%s,%f,%s,%s\n" n s.iters s.end_res/s.st_res s.cyclecomplexity*s.iters wda(s.cyclecomplexity, s.end_res/s.st_res, s.iters) s.setup_time s.solve_time relax ci agg mr
                write(dump_file, s)
            end
        end
    end
end

# replace this with your matrix loading code
using MAT
L=matread(expanduser(ARGS[1]))["L"]

s = ArgParseSettings()
@add_arg_table s begin
    "--dump"
        arg_type = String
        help = "Dump outputs to csv file"
    "MAT"
        help = "Graph Laplacian file"
        required = true
end

parsed_args = parse_args(ARGS, s)

dump_file = nothing
if !isa(parsed_args["dump"], Void)
    dump_file = open(parsed_args["dump"], "w")
    write(dump_file, "solver,iters,rel_error,work,wda,setup_time,solve_time,relaxation,cycle_index,aggregation,minres\n")
end

# L should be a graph lacplacian (not an adjacency matrix)
test_solvers(L, dump_file)
