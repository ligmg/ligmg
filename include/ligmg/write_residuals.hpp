#pragma once

#include <boost/multi_array.hpp>
#include <h5xx/h5xx.hpp>
#include "util.hpp"

namespace ligmg {
template <class T, class F, class DER>
void write_sparse_mat(h5xx::group& g, const std::string name,
                      const combblas::SpParMat<T, F, DER>& A) {
  h5xx::group gr;
  if (A.getcommgrid()->GetRank() == 0) gr = h5xx::group(g, name);
  std::cout << 1 << std::endl;
  auto t = gather_ijk(A);
  std::cout << "here" << std::endl;
  if (A.getcommgrid()->GetRank() == 0) {
    h5xx::create_dataset(gr, "is", std::get<0>(t));
    std::cout << 1 << std::endl;
    h5xx::write_dataset(gr, "is", std::get<0>(t));
    std::cout << 2 << std::endl;
    h5xx::create_dataset(gr, "js", std::get<1>(t));
    std::cout << 3 << std::endl;
    h5xx::write_dataset(gr, "js", std::get<1>(t));
    std::cout << 4 << std::endl;
    h5xx::create_dataset(gr, "ks", std::get<2>(t));
    std::cout << 5 << std::endl;
    h5xx::write_dataset(gr, "ks", std::get<2>(t));
    std::cout << 6 << std::endl;
  }
}

template <class T, class F, class DER>
void write_mat(const std::string filename, const std::string name,
               const combblas::SpParMat<T, F, DER>& A) {
  bool o = A.getcommgrid()->GetRank() == 0;
  h5xx::file file;
  if (o) file = h5xx::file(filename, h5xx::file::trunc);
  h5xx::group root;
  if (o) root = h5xx::group(file);
  write_sparse_mat(root, name, A);
}

/*
 * HDF5 file formatted as follows:
 * { iters: 2
 * , levels: 3
 * , (iter #):
 *      { (level #): (# verts x # steps) }
 * }
 */
// template <class T, class F, class DER, class Vec>
// void write_debug_info(const std::vector<Level<T, F, DER, Vec>>& levels,
//                       const DebugInfo<F>& db, std::string filename) {
//   bool o = levels[0].A.getcommgrid()->GetRank() == 0;
//   if (o) std::cout << "Dumping debug info..." << std::flush;
//
//   h5xx::file file;
//   if (o) file = h5xx::file(filename, h5xx::file::trunc);
//   h5xx::group root;
//   if (o) root = h5xx::group(file);
//   h5xx::group res;
//   if (o) res = h5xx::group(file, "residuals");
//
//   // write # of iterations
//   if (o) h5xx::write_attribute(res, "iters", db.residuals.size());
//
//   // write # of levels
//   if (o) h5xx::write_attribute(res, "levels", levels.size());
//
//   for (int i = 0; i < db.residuals.size(); i++) {
//     // create iteration group
//     h5xx::group g;
//     if (o) g = h5xx::group(res, "iter" + std::to_string(i));
//
//     // write each level
//     if (o)
//       for (int j = 0; j < db.residuals[i].size(); j++) {
//         auto& res = db.residuals[i][j];
//         int sx = res.size();
//         if (sx == 0) continue;
//
//         int sy = res[0].size();
//         // std::vector<F> array(sy*sx);
//         boost::multi_array<F, 2> array(boost::extents[sy][sx]);
//         for (int x = 0; x < res.size(); x++) {
//           for (int y = 0; y < res[x].size(); y++) {
//             array[y][x] = res[x][y];
//             // array[sx * y + x] = res[x][y];
//           }
//         }
//         h5xx::create_dataset(g, "res" + std::to_string(j), array);
//         h5xx::write_dataset(g, "res" + std::to_string(j), array);
//       }
//   }
//
//   // store each levels matrices
//   h5xx::group mats;
//   if (o) mats = h5xx::group(root, "mats");
//   for (int i = 0; i < levels.size(); i++) {
//     // new group for level
//     h5xx::group lvl;
//     if (o) lvl = h5xx::group(mats, std::to_string(i));
//
//     // store level type
//     std::string type;
//     if (i == levels.size() - 1) {
//       type = "coarse";
//     } else if (levels[i].is_elim) {
//       type = "elimination";
//     } else {
//       type = "aggregation";
//     }
//     if (o) h5xx::write_attribute(lvl, "type", type);
//
//     // write A matrix
//     write_sparse_mat(lvl, "A", levels[i].A);
//
//     if (i == levels.size()) {
//     } else if (levels[i].is_elim) {
//       write_sparse_mat(lvl, "Q", levels[i].Q);
//       write_sparse_mat(lvl, "R", levels[i].R);
//       write_sparse_mat(lvl, "P", levels[i].P);
//     } else {
//       write_sparse_mat(lvl, "R", levels[i].R);
//       write_sparse_mat(lvl, "P", levels[i].P);
//     }
//   }
//
//   if (o) std::cout << "done" << std::endl;
// }
}
