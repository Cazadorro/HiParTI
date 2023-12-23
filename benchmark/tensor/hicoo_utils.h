//
// Created by User on 11/3/2023.
//

#ifndef HIPARTI_HICOO_UTILS_H
#define HIPARTI_HICOO_UTILS_H

#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../src/sptensor/hicoo/hicoo.h"
#include <iostream>
#include <sstream>

namespace util {
//    typedef struct {
//        ptiIndex nmodes;      /// # modes
//        ptiIndex * sortorder;  /// the order in which the indices are sorted
//        ptiIndex * ndims;      /// size of each mode, length nmodes
//        ptiNnzIndex nnz;         /// # non-zeros
//        ptiIndexVector * inds;       /// indices of each element, length [nmodes][nnz]
//        ptiValueVector values;      /// non-zero values, length nnz
//    } ptiSparseTensor;
    inline std::string print_sparse(const ptiSparseTensor& tensor) {
        std::stringstream ss;
        ss << "nmodes : " << tensor.nmodes << "\n";
        ss << "sortorder : ";
        for (std::size_t i = 0; i < tensor.nmodes; ++i) {
            ss << tensor.sortorder[i] << ", ";
        }
        ss << "\n";
        ss << "ndims : ";
        for (std::size_t i = 0; i < tensor.nmodes; ++i) {
            ss << tensor.ndims[i] << ", ";
        }
        ss << "\n";
        ss << "nnz : " << tensor.nnz << "\n";
        for(std::size_t i = 0; i < tensor.nmodes; ++i){
            ss << "Indices for mode : " << i << "\n";
            ss << "len : " << tensor.inds[i].len << "\n";
            ss << "capacity : " << tensor.inds[i].cap << "\n";
            for(std::size_t j = 0; j < tensor.nnz; ++j){
                ss << "index: " << tensor.inds[i].data[j] << "\n";
            }
        }
        ss << "\n";
        ss << "data\n";
        ss << "len : " << tensor.values.len << "\n";
        ss << "capacity : " << tensor.values.cap << "\n";
        for(std::size_t j = 0; j < tensor.nnz; ++j){
            for(std::size_t i = 0; i < tensor.nmodes; ++i){
                ss << tensor.inds[i].data[j] << ", ";
            }
            ss << ": " << tensor.values.data[j] << "\n";
        }
        return ss.str();
    }
}

#endif //HIPARTI_HICOO_UTILS_H
