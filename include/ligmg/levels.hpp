#pragma once

#include <functional>
#include "options.hpp"

namespace ligmg {

template <class T, class F, class DER>
struct Level {
  virtual void do_level(
      combblas::DenseParVec<T, F>& x  // current guess at the solution
      ,
      const combblas::DenseParVec<T, F>& rhs  // right hand side to solve for
      ,
      std::function<void(combblas::DenseParVec<T, F>&, const combblas::DenseParVec<T, F>&)>
          recurse  // function to recurse to next level
      ,
      Options opts) = 0;
  // indicate that this level exactly interpolates solutions
  virtual bool is_exact() = 0;
  virtual combblas::SpParMat<T,F,DER>& matrix() = 0;
  virtual T nnz() = 0;
  virtual T size() = 0;
  virtual T work(Options) = 0;
  virtual std::string name() = 0;
  virtual std::shared_ptr<combblas::CommGrid> commgrid() = 0;
  virtual void dump(std::string) = 0;
};
}
