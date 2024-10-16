//
// Created by User on 11/3/2023.
//

#ifndef HIPARTI_HICOO_UTILS_H
#define HIPARTI_HICOO_UTILS_H

#include <czdr/bitutil/bit_vector.h>
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../include/includes/sptensors.h"
#include "../src/sptensor/hicoo/hicoo.h"
#include "../include/includes/structs.h"
#include <iostream>
#include <sstream>
#include <csrk.h>

namespace util {

//    class Transpose2DBitfield{
//    public:
//        Transpose2DBitfield(std::size_t width);
//        bool test_and_set(std::size_t row, std::size_t col);
//    private:
//        std::size_t m_width;
//        czdr::bit_vector<std::uint32_t> m_bit_field;
//
//    };

    ptiSparseTensor transpose(const ptiSparseTensor * tsr);
    ptiSparseTensor transpose_add(const ptiSparseTensor * tsr);

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


    struct CSRTensor{
        std::int32_t rows;
        std::int32_t cols;
        std::int32_t nnz;
        std::vector<std::uint32_t> row_start;
        std::vector<std::uint32_t> col_idxs;
        std::vector<float> values;
    };

    inline CSRTensor coo_to_csr(ptiSparseTensor* coo_tensor){
        //TODO assumes coo_tensor sorted, assumes y dim (row dim) is the first set of indices.
        // Also assumes 2D.
        CSRTensor csr_tensor;
        csr_tensor.rows = coo_tensor->ndims[0];
        csr_tensor.cols = coo_tensor->ndims[1];
        csr_tensor.nnz = coo_tensor->nnz;

        csr_tensor.row_start.push_back(0);
        std::size_t previous_row_idx = coo_tensor->inds[0].data[0];
        csr_tensor.col_idxs.reserve(csr_tensor.nnz);
        csr_tensor.values.reserve(csr_tensor.nnz);
        for(std::size_t idx = 0; idx < csr_tensor.nnz; ++idx){
            csr_tensor.col_idxs.push_back(coo_tensor->inds[1].data[idx]);
            csr_tensor.values.push_back(coo_tensor->values.data[idx]);
            if(previous_row_idx != coo_tensor->inds[0].data[idx]){
                csr_tensor.row_start.push_back(idx);
                previous_row_idx = coo_tensor->inds[0].data[idx];
            }
        }
        //todo don't know if we need this.
        csr_tensor.row_start.push_back(csr_tensor.nnz);
        return std::move(csr_tensor);
    }


    inline CSRk_Graph coo_to_csrk(ptiSparseTensor* coo_tensor){

    }





}

#endif //HIPARTI_HICOO_UTILS_H
