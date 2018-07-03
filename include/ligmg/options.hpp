#pragma once

#include <iostream>
#include <mxx/comm.hpp>
#include <petsc.h>
#include <string>

namespace ligmg {

enum class AffScaling { None, Row, Column, RowColumn, Sym, Max };

enum class SmootherType { Jacobi, Chebyshev };

enum class NormType { Norm2, NormInf, NormBoth };

std::ostream &operator<<(std::ostream &os, SmootherType a) {
  switch (a) {
  case SmootherType::Jacobi:
    os << "jacobi";
    break;
  case SmootherType::Chebyshev:
    os << "chebyshev";
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, AffScaling a) {
  if (a == AffScaling::None) {
    os << "None";
  } else if (a == AffScaling::Row) {
    os << "combblas::Row";
  } else if (a == AffScaling::Column) {
    os << "Column";
  } else if (a == AffScaling::RowColumn) {
    os << "RowColumn";
  } else if (a == AffScaling::None) {
    os << "None";
  } else if (a == AffScaling::Max) {
    os << "Max";
  } else if (a == AffScaling::Sym) {
    os << "Sym";
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, NormType a) {
  switch (a) {
  case NormType::Norm2:
    os << "2";
    break;
  case NormType::NormInf:
    os << "inf";
    break;
  case NormType::NormBoth:
    os << "inf";
    break;
  }
  return os;
}

enum class StrengthMetric { Affinity, AlgebraicDistance, None };

std::ostream &operator<<(std::ostream &os, StrengthMetric a) {
  switch (a) {
  case StrengthMetric::Affinity:
    os << "affinity";
    break;
  case StrengthMetric::AlgebraicDistance:
    os << "algebraic-distance";
    break;
  case StrengthMetric::None:
    os << "none";
    break;
  }
  return os;
}

// TODO: split options into various parts
struct Options {
  bool verbose{false};
  bool verbose_solve{false};
  bool redist{true};
  bool always_elim{true};
  bool top_elim{true};
  std::string dump{""};
  bool constant_correction{false};
  std::string csv{""};
  StrengthMetric strength_metric{StrengthMetric::Affinity};
  bool test_vcycle{false};
  int max_levels{50};
  int64_t min_size{4000};
  int elim_degree{4};
  double tol{1e-8};
  double thresh{1e-8};
  int agg_iters{10};
  int iters{500};
  double redist_factor{10};
  double redist_bound{10000};
  int pre_smooth{2};
  int post_smooth{2};
  int num_tvs{10};
  int tvs_smooth{4};
  bool test_jacobi{false};
  SmootherType smoother_type{SmootherType::Chebyshev};
  std::string load_file{""};
  bool kcycle{false};
  bool test_cg{false};
  KSPType outer_ksp_type{KSPCG};
  KSPType kcycle_ksp_type{KSPFCG};
  int kcycle_iters{2};
  int64_t votes_to_seed{1};
  int64_t max_agg_size{-1};
  double agg_iter_factor{0.5};
  double agg_init_factor{0.5};
  double cheby_lower{0.3};
  double cheby_upper{1.1};
  AffScaling aff_scaling{AffScaling::Max};
  bool rand_perm{true};
  bool adj{false};
  double elim_factor{0.9};
  bool cc{false};
  int elim_iters{1};
  double rejection_ratio{-1};
  double cr_threshold{0.1};
  int cr_tvs{4};
  int cr_iters{2};
  int cr_smoothing_iters{4};
  NormType cr_norm{NormType::NormBoth};
  bool neighbor_of_neighbor{false};
  bool rejection_to_seed{false};
  bool test_ligmg{true};
  std::string rhs{""};
};

Options default_options() { return Options(); }


void print_options_csv_header(std::ostream &os) {
  os << "StrengthMetric,Redistribute,MaxLevels,MinSize,ElimDegree,AlwaysElim,"
        "TopElim,"
        "Tolerance,Threshold,AggregationIterations,MaxIterations,"
        "RedistributionFactor,RedistributionBound,PreSmoothing,PostSmoothing,"
        "ConstantCorrection,Smoother,NumberTVs,TVSmoothingIterations,KCycle,"
        "KCycleIterations,OuterKSPType,KCycleKSPType,VotesToSeed,MaxAggSize,"
        "AggIterFactor,ChebyLower,ChebyUpper,AffScaling,RandPerm,"
        "RejectionRatio,CRThreshold,CRTVs,CRIters,CRSmoothingIters,CRNorm";
}

std::ostream &operator<<(std::ostream &os, const Options &opts) {
  os << opts.strength_metric << ',' << opts.redist << ',' << opts.max_levels
     << ',' << opts.min_size << ',' << opts.elim_degree << ','
     << opts.always_elim << ',' << opts.top_elim << ',' << opts.tol << ','
     << opts.thresh << ',' << opts.agg_iters << ',' << opts.iters << ','
     << opts.redist_factor << ',' << opts.redist_bound << ',' << opts.pre_smooth
     << ',' << opts.post_smooth << ',' << opts.constant_correction << ','
     << opts.smoother_type << ',' << opts.num_tvs << ',' << opts.tvs_smooth
     << ',' << opts.kcycle << ',' << opts.kcycle_iters << ','
     << opts.outer_ksp_type << ',' << opts.kcycle_ksp_type << ','
     << opts.votes_to_seed << ',' << opts.max_agg_size << ','
     << opts.agg_iter_factor << ',' << opts.cheby_lower << ','
     << opts.cheby_upper << ',' << opts.aff_scaling << ',' << opts.rand_perm
     << ',' << opts.rejection_ratio << ',' << opts.cr_threshold << ','
     << opts.cr_tvs << ',' << opts.cr_iters << ',' << opts.cr_smoothing_iters
     << ',' << opts.cr_norm;

  return os;
}
}
