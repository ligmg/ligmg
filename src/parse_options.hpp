#pragma once

#include <ligmg/options.hpp>
#include "OptionParser.h"

ligmg::Options parse_options(int argc, char *argv[]) {
  using namespace ligmg;
  ligmg::Options def = ligmg::default_options();
  // parse arguments
  optparse::OptionParser parser = optparse::OptionParser().description(
      "A distributed memory graph Laplacian solver");

  parser.add_option("-v", "--verbose")
      .dest("verbose")
      .help("print debugging output")
      .action("store_true")
      .set_default(def.verbose);
  parser.add_option("--verbose-solve")
      .dest("verbose-solve")
      .help("print solve debugging output")
      .action("store_true")
      .set_default(def.verbose_solve);
  parser.add_option("--strength")
      .dest("strength-metric")
      .set_default(def.strength_metric)
      .help("Which strength of connection metric to use")
      .action("store");
  parser.add_option("--test-vcycle")
      .dest("test-vcycle")
      .set_default(def.test_vcycle)
      .help("Measure plain vcycle performance")
      .action("store_true");
  parser.add_option("--no-redist")
      .dest("redist")
      .help("redistribute multigrid hierarchy to keep work balanced")
      .action("store_false")
      .set_default(def.redist);
  parser.add_option("--max-levels")
      .dest("max-levels")
      .help("maximum levels in multigrid hierarchy")
      .set_default(def.max_levels)
      .type("int")
      .action("store");
  parser.add_option("--min-size")
      .dest("min-size")
      .help("minimum size (nnz) of coarsest level in multigrid hierarchy")
      .set_default(def.min_size)
      .type("int")
      .action("store");
  parser.add_option("--elim-degree")
      .dest("elim-degree")
      .help("maximum degree to eliminate")
      .set_default(def.elim_degree)
      .type("int")
      .action("store");
  parser.add_option("--never-elim")
      .dest("always-elim")
      .help("don't perform elimination on every level")
      .action("store_false")
      .set_default(def.always_elim);
  parser.add_option("--always-top-elim")
      .dest("top-elim")
      .help(
          "apply top level elimination every vcycle (otherwise performed only "
          "once)")
      .action("store_false")
      .set_default(def.top_elim);
  parser.add_option("--tol")
      .dest("tol")
      .help("stopping relative tolerance")
      .action("store")
      .type("double")
      .set_default(def.tol);
  parser.add_option("--thresh")
      .dest("thresh")
      .help("Affinity cutoff threshold")
      .action("store")
      .type("double")
      .set_default(def.thresh);
  parser.add_option("--agg-iters")
      .dest("agg-iters")
      .help("number of aggregation iterations")
      .action("store")
      .type("int")
      .set_default(def.agg_iters);
  parser.add_option("--iters")
      .dest("iters")
      .help("number of solver iterations")
      .action("store")
      .type("int")
      .set_default(def.iters);
  parser.add_option("--redist-factor")
      .dest("redist-factor")
      .help("factor for redistributing work")
      .action("store")
      .type("double")
      .set_default(def.redist_factor);
  parser.add_option("--redist-bound")
      .dest("redist-bound")
      .help("Bound for redistributing work")
      .action("store")
      .type("double")
      .set_default(def.redist_bound);
  parser.add_option("--pre-iters")
      .dest("pre-iters")
      .help("number of pre-restriction smoothing iterations")
      .action("store")
      .type("int")
      .set_default(def.pre_smooth);
  parser.add_option("--post-iters")
      .dest("post-iters")
      .help("number of post-restriction smoothing iterations")
      .action("store")
      .type("int")
      .set_default(def.post_smooth);
  parser.add_option("--dump").dest("dump").metavar("FILE").help(
      "Dump hierarchy and residual information to FILE");
  parser.add_option("--csv")
      .dest("csv")
      .metavar("FILENAME")
      .set_default("")
      .help("Output stats to csv as FILENAME");
  parser.add_option("--constant-correction")
      .dest("constant-correction")
      .action("store_true")
      .set_default(def.constant_correction)
      .help("Apply 4/3 constant correction");
  parser.add_option("--test-jacobi")
      .dest("test-jacobi")
      .action("store_true")
      .set_default(def.test_jacobi)
      .help("Test jacobi convergence");
  parser.add_option("--test-cg")
      .dest("test-cg")
      .action("store_true")
      .set_default(def.test_cg)
      .help("Test CG convergence");
  parser.add_option("--jacobi-smoothing")
      .dest("jacobi-smoothing")
      .action("store_true")
      .set_default(false)
      .help("Use Jacobi smoothing (default is chebyshev)");
  parser.add_option("--num-tvs")
      .dest("num-tvs")
      .help("Number of test vectors")
      .action("store")
      .type("int")
      .set_default(def.num_tvs);
  parser.add_option("--tvs-smooth")
      .dest("tvs-smooth")
      .help("Number of smoothing iterations on test vectors")
      .action("store")
      .type("int")
      .set_default(def.tvs_smooth);
  parser.add_option("--kcycle")
      .dest("kcycle")
      .action("store_true")
      .set_default(def.kcycle)
      .help("Use K-cycles");
  parser.add_option("--kcycle-iters")
      .dest("kcycle-iters")
      .set_default(def.kcycle_iters)
      .help("Number of iterations per kcycle");
  parser.add_option("--kcycle-ksp-type")
      .dest("kcycle-ksp-type")
      .metavar("KSPTYPE")
      .set_default(def.kcycle_ksp_type)
      .help("Type of PETSc CG method to use in kcycles (pipefcg, fcg, fgmres "
            ")");
  parser.add_option("--outer-ksp-type")
      .dest("outer-ksp-type")
      .metavar("KSPTYPE")
      .set_default(def.outer_ksp_type)
      .help(
          "Type of PETSc CG method to use for outer accelerator (pipefcg, fcg, "
          "cg, fgmres)");
  parser.add_option("--votes-to-seed")
      .dest("votes-to-seed")
      .help("Number of votes to turn an undecided node into a seed")
      .action("store")
      .type("double")
      .set_default(def.votes_to_seed);
  parser.add_option("--max-agg-size")
      .dest("max-agg-size")
      .help(
          "Maximum size of an aggregate (not a hard limit). Use 0 for no size "
          "limit.")
      .action("store")
      .type("int")
      .set_default(def.max_agg_size);
  parser.add_option("--agg-iter-factor")
      .dest("agg-iter-factor")
      .help("Reduction factor of connection strength in aggregation routine. "
            "Must be in the range 1 > factor > 0")
      .action("store")
      .type("double")
      .set_default(def.agg_iter_factor);
  parser.add_option("--agg-init-factor")
      .dest("agg-init-factor")
      .help("Initial reduction factor of connection strength in aggregation "
            "routine. "
            "Must be in the range 1 > factor > 0")
      .action("store")
      .type("double")
      .set_default(def.agg_init_factor);
  parser.add_option("--cheby-lower")
      .dest("cheby-lower")
      .help("Fraction of largest eigenvalue used for Chebyshev lower bound")
      .action("store")
      .type("double")
      .set_default(def.cheby_lower);
  parser.add_option("--cheby-upper")
      .dest("cheby-upper")
      .help("Fraction of largest eigenvalue used for Chebyshev upper bound")
      .action("store")
      .type("double")
      .set_default(def.cheby_upper);
  parser.add_option("--aff-scaling")
      .dest("aff-scaling")
      .help("How to scale the affinity matrix (none, row, column, rowcolumn, "
            "sym)")
      .action("store")
      .type("string")
      .set_default("max");
  parser.add_option("--no-rand-perm")
      .dest("rand-perm")
      .action("store_false")
      .set_default(def.rand_perm)
      .help("Randomly permute matrix for load balancing");
  parser.add_option("--adj")
      .dest("adj")
      .action("store_true")
      .set_default(def.adj)
      .help("Input is an adjacency matrix");
  parser.add_option("--elim-factor")
      .dest("elim-factor")
      .help("Fraction of the graph that needs to be eliminated in order to add "
            "elimination level")
      .action("store")
      .type("double")
      .set_default(def.elim_factor);
  parser.add_option("--cc")
      .dest("cc")
      .action("store_true")
      .set_default(def.cc)
      .help("Run connected components");
  parser.add_option("--elim-iters")
      .dest("elim-iters")
      .help("Number of iterations of partial elimination")
      .action("store")
      .type("int")
      .set_default(def.elim_iters);
  parser.add_option("--rejection-ratio")
      .dest("rejection-ratio")
      .help("Threshold for rejecting vertices from an aggregate based on the "
            "ratio of edge weight sums internally vs externally.")
      .action("store")
      .type("double")
      .set_default(def.rejection_ratio);
  parser.add_option("--cr-threshold")
      .dest("cr-threshold")
      .help("Threshold for rejecting and aggregate based on norm of CR test "
            "vectors.")
      .action("store")
      .type("double")
      .set_default(def.cr_threshold);
  parser.add_option("--cr-tvs")
      .dest("cr-tvs")
      .help("Number of test vectors to use for compatible relaxation")
      .action("store")
      .type("int")
      .set_default(def.cr_tvs);
  parser.add_option("--cr-iters")
      .dest("cr-iters")
      .help("Number of compatible relaxation iterations")
      .action("store")
      .type("int")
      .set_default(def.cr_iters);
  parser.add_option("--cr-smoothing-iters")
      .dest("cr-smoothing-iters")
      .help("Number of smoothing iterations to use for compatible relaxation")
      .action("store")
      .type("int")
      .set_default(def.cr_smoothing_iters);
  parser.add_option("--cr-norm")
      .dest("cr-norm")
      .help("Norm to determine if aggregate is bad")
      .action("store")
      .type("string")
      .set_default("both");
  parser.add_option("--neighbor-of-neighbor")
      .dest("neighbor-of-neighbor")
      .help("Aggregate neighbors of neighbors of the root")
      .action("store_true")
      .set_default(def.neighbor_of_neighbor);
  parser.add_option("--rejection-to-seed")
      .dest("rejection-to-seed")
      .help("Rejected nodes become seeds")
      .action("store_true")
      .set_default(def.rejection_to_seed);
  parser.add_option("--no-test-ligmg")
      .dest("test-ligmg")
      .help("Test LIGMG performance")
      .action("store_false")
      .set_default(def.test_ligmg);
  parser.add_option("--rhs")
      .dest("rhs")
      .help("Right hand side to solve against. Random RHS is used if this option is not specified")
      .set_default(def.rhs);

  optparse::Values options = parser.parse_args(argc, argv);
  std::vector<std::string> args = parser.args();
  if (args.size() != 1) {
    if (mxx::comm(MPI_COMM_WORLD).rank() == 0)
      std::cout << "Must provide matrix" << std::endl;
    exit(1);
  }

  ligmg::Options opts;
  opts.verbose = options.get("verbose");
  opts.verbose_solve = options.get("verbose-solve");
  opts.redist = options.get("redist");
  opts.always_elim = options.get("always-elim");
  opts.top_elim = options.get("top-elim");
  opts.dump = options["dump"];
  opts.csv = std::string(options.get("csv"));
  opts.test_vcycle = options.get("test-vcycle");
  opts.constant_correction = options.get("constant-correction");
  opts.max_levels = int(options.get("max-levels"));
  opts.min_size = int(options.get("min-size"));
  opts.elim_degree = int(options.get("elim-degree"));
  opts.tol = double(options.get("tol"));
  opts.thresh = double(options.get("thresh"));
  opts.agg_iters = int(options.get("agg-iters"));
  opts.iters = int(options.get("iters"));
  opts.redist_factor = double(options.get("redist-factor"));
  opts.redist_bound = double(options.get("redist-bound"));
  opts.pre_smooth = int(options.get("pre-iters"));
  opts.post_smooth = int(options.get("post-iters"));
  opts.num_tvs = int(options.get("num-tvs"));
  opts.tvs_smooth = int(options.get("tvs-smooth"));
  opts.test_jacobi = options.get("test-jacobi");
  opts.test_cg = options.get("test-cg");
  opts.kcycle = options.get("kcycle");
  opts.kcycle_iters = int(options.get("kcycle-iters"));
  opts.votes_to_seed = double(options.get("votes-to-seed"));
  opts.max_agg_size = int(options.get("max-agg-size"));
  opts.agg_iter_factor = double(options.get("agg-iter-factor"));
  opts.agg_init_factor = double(options.get("agg-init-factor"));
  opts.cheby_lower = double(options.get("cheby-lower"));
  opts.cheby_upper = double(options.get("cheby-upper"));
  opts.rand_perm = options.get("rand-perm");
  opts.adj = options.get("adj");
  opts.elim_factor = double(options.get("elim-factor"));
  opts.cc = options.get("cc");
  opts.elim_iters = int(options.get("elim-iters"));
  opts.rejection_ratio = double(options.get("rejection-ratio"));
  opts.cr_threshold = double(options.get("cr-threshold"));
  opts.cr_tvs = int(options.get("cr-tvs"));
  opts.cr_iters = int(options.get("cr-iters"));
  opts.cr_smoothing_iters = int(options.get("cr-smoothing-iters"));
  opts.neighbor_of_neighbor = options.get("neighbor-of-neighbor");
  opts.rejection_to_seed = options.get("rejection-to-seed");
  opts.test_ligmg = options.get("test-ligmg");
  opts.rhs = std::string(options.get("rhs"));

  if (bool(options.get("jacobi-smoothing"))) {
    opts.smoother_type = SmootherType::Jacobi;
  } else {
    opts.smoother_type = SmootherType::Chebyshev;
  }

  std::string ksptype = std::string(options.get("outer-ksp-type"));
  if(opts.kcycle && !options.is_set_by_user("outer-ksp-type")) {
    opts.outer_ksp_type = KSPFCG;
  } else {
    if (ksptype == "fcg") {
      opts.outer_ksp_type = KSPFCG;
    } else if (ksptype == "cg") {
      opts.outer_ksp_type = KSPCG;
    } else if (ksptype == "pipefcg") {
      opts.outer_ksp_type = KSPPIPEFCG;
    } else if (ksptype == "fgmres") {
      opts.outer_ksp_type = KSPFGMRES;
    } else {
      std::cout
        << "Invalid option for outer-ksp-type. Must be one of pipefcg, fcg, cg "
        "fgmres"
        << std::endl;
      exit(1);
    }
  }

  ksptype = std::string(options.get("kcycle-ksp-type"));
  if (ksptype == "fcg") {
    opts.kcycle_ksp_type = KSPFCG;
  } else if (ksptype == "pipefcg") {
    opts.kcycle_ksp_type = KSPPIPEFCG;
  } else if (ksptype == "fgmres") {
    opts.kcycle_ksp_type = KSPFGMRES;
  } else {
    std::cout
        << "Invalid option for kcycle-ksp-type. Must be one of pipefcg, fcg, "
           "fgmres"
        << std::endl;
    exit(1);
  }
  std::string affscaling = std::string(options.get("aff-scaling"));
  if (affscaling == "none") {
    opts.aff_scaling = AffScaling::None;
  } else if (affscaling == "row") {
    opts.aff_scaling = AffScaling::Row;
  } else if (affscaling == "column") {
    opts.aff_scaling = AffScaling::Column;
  } else if (affscaling == "rowcolumn") {
    opts.aff_scaling = AffScaling::RowColumn;
  } else if (affscaling == "sym") {
    opts.aff_scaling = AffScaling::Sym;
  } else if (affscaling == "max") {
    opts.aff_scaling = AffScaling::Max;
  } else {
    std::cout << "Invalid option for aff-scaling. Must be one of none, row, "
                 "column, rowcolumn, sym, max"
              << std::endl;
    exit(1);
  }

  std::string strength_metric = std::string(options.get("strength-metric"));
  if (strength_metric == "affinity") {
    opts.strength_metric = StrengthMetric::Affinity;
  } else if (strength_metric == "algebraic-distance") {
    opts.strength_metric = StrengthMetric::AlgebraicDistance;
  } else if (strength_metric == "none") {
    opts.strength_metric = StrengthMetric::None;
  }

  std::string normtype = std::string(options.get("cr-norm"));
  if (normtype == "2") {
    opts.cr_norm = ligmg::NormType::Norm2;
  } else if (normtype == "inf") {
    opts.cr_norm = ligmg::NormType::NormInf;
  } else if (normtype == "both") {
    opts.cr_norm = ligmg::NormType::NormBoth;
  } else {
    std::cout << "Invalid option for cr-norm. Must be one of 2, inf, both"
              << std::endl;
    exit(1);
  }

  opts.load_file = args[0];

  return opts;
}
