#pragma once

#include <CombBLAS/CombBLAS.h>
#include <mxx/collective.hpp>
#include <iostream>

namespace combblas {

  // Create a vector filled with the given constant
  template<class T, class F, class DER>
  combblas::DenseParVec<T,F> const_vec(const combblas::SpParMat<T,F,DER>& A, F constant) {
    return combblas::DenseParVec<T, F>(A.getcommgrid(), A.seq().getncol(), constant, 0);
  }

  // Create a std::vector filled with random values between 0 and 1
  template<class T, class F, class DER>
  combblas::DenseParVec<T,F> rand_vec(const combblas::SpParMat<T,F,DER>& A) {
    combblas::DenseParVec<T, F> x(A.getcommgrid(), A.seq().getncol(), 0, 0);
    for (auto& y : x.data()) {
      y = F(rand()) / RAND_MAX;
    }
    return x;
  }


  // Write a DenseParVec to a file in petsc binary format
  template<class T, class F>
  void write_petsc(const combblas::DenseParVec<T,F>& vec, std::string filename) {
    int classid = __builtin_bswap32(1211214);

    auto v = mxx::gatherv(vec.data(), 0, vec.getcommgrid()->GetWorld());

    if(vec.getcommgrid()->GetRank() == 0) {
      std::ofstream of(filename);
      of.write((char*)&classid, sizeof(int));
      int len = vec.data().size();
      int len_ = __builtin_bswap32(len);
      of.write((char*)&len_, sizeof(int));
      for(auto f : vec.data()) {
        uint64_t x = __builtin_bswap64(*(uint64_t*)&f);
        of.write((char*)&x, sizeof(F));
      }
    }
  }
}
