//
// Created by User on 11/3/2023.
//

#include "hicoo_utils.h"
#include <iterator>
#include <algorithm>
#include <compare>
namespace util {
    ptiSparseTensor transpose(const ptiSparseTensor *tsr) {
        ptiSparseTensor tsr_t;
        ptiCopySparseTensor(&tsr_t, tsr, 4);
        std::vector<ptiIndex> sorted_indexes(tsr->nmodes);
        for (std::size_t i = 0; i < tsr_t.nnz; ++i) {
            for (std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++dim_idx) {
                sorted_indexes[dim_idx] = tsr_t.inds[dim_idx].data[i];
            }
            std::reverse(sorted_indexes.begin(), sorted_indexes.end());
            for (std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++dim_idx) {
                tsr_t.inds[dim_idx].data[i] = sorted_indexes[dim_idx];
            }
        }
        //reverse order of dimension sizes as well.
        std::reverse(tsr_t.ndims, tsr_t.ndims + tsr_t.nmodes);
        ptiSparseTensorSortIndex(&tsr_t, 1, 1);
        return tsr_t;
    }

    std::weak_ordering compare_index(const ptiSparseTensor &lhs, const ptiSparseTensor &rhs, std::size_t lhs_idx, std::size_t rhs_idx){
        assert(lhs.nmodes == rhs.nmodes);
        assert(lhs.nnz == rhs.nnz);
        for (std::size_t dim_idx = 0; dim_idx < lhs.nmodes; ++dim_idx) {
            auto lhs_index_value = lhs.inds[dim_idx].data[lhs_idx];
            auto rhs_index_value = rhs.inds[dim_idx].data[rhs_idx];
            if(lhs_index_value < rhs_index_value){
                return std::weak_ordering::less;
            }
            if(lhs_index_value > rhs_index_value){
                return std::weak_ordering::greater;
            }
        }
        return std::weak_ordering::equivalent;
    }


    int newSparseTensor(ptiSparseTensor *tsr, ptiIndex nmodes, const ptiIndex ndims[],
                         const std::vector<ptiValue>& values,
                         const std::vector<std::vector<ptiIndex>>& indices) {
        ptiIndex i;
        int result;
        tsr->nmodes = nmodes;
        tsr->sortorder = reinterpret_cast<ptiIndex *>(malloc(nmodes * sizeof tsr->sortorder[0]));
        for(i = 0; i < nmodes; ++i) {
            tsr->sortorder[i] = i;
        }
        tsr->ndims = reinterpret_cast<ptiIndex *>(malloc(nmodes * sizeof *tsr->ndims));
        pti_CheckOSError(!tsr->ndims, "SpTns New");
        memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
        tsr->nnz = 0;
        tsr->inds = reinterpret_cast<ptiIndexVector *>(malloc(nmodes * sizeof *tsr->inds));
        pti_CheckOSError(!tsr->inds, "SpTns New");
        for(i = 0; i < nmodes; ++i) {

            result = ptiNewIndexVector(&tsr->inds[i], values.size(),  values.size());
            pti_CheckError(result, "SpTns New", NULL);
            std::copy(indices[i].begin(), indices[i].end(), tsr->inds[i].data);
        }
        result = ptiNewValueVector(&tsr->values, values.size(), values.size());
        pti_CheckError(result, "SpTns New", NULL);
        std::copy(values.begin(), values.end(), tsr->values.data);
        return 0;
    }

    ptiSparseTensor transpose_add(const ptiSparseTensor *tsr) {
        ptiSparseTensor tsr_t = transpose(tsr);
        ptiSparseTensor tsr_add;
        std::vector<ptiValue> merge_value_list;
        merge_value_list.reserve(tsr->nnz * 2);
        std::vector<std::vector<ptiIndex>> merge_index_list(tsr->nmodes);
        std::vector<ptiIndex> ndims(tsr->nmodes);
        for(std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++dim_idx){
            auto max_dim = std::max(tsr->ndims[dim_idx], tsr_t.ndims[dim_idx]);
            merge_index_list[dim_idx].reserve(max_dim);
            ndims.push_back(max_dim);
        }


        std::size_t lhs_idx = 0;
        std::size_t rhs_idx = 0;
        while(lhs_idx < tsr->nnz || rhs_idx < tsr_t.nnz ){
            auto ordering = compare_index(*tsr, tsr_t, lhs_idx, rhs_idx);
            if(ordering == std::weak_ordering::equivalent){
                merge_value_list.push_back(tsr->values.data[lhs_idx] + tsr_t.values.data[lhs_idx]);
                for(std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++dim_idx){
                    merge_index_list[dim_idx].push_back(tsr->inds[dim_idx].data[lhs_idx]);
                }
                lhs_idx += 1;
                rhs_idx += 1;
            }else if(ordering == std::weak_ordering::less){
                merge_value_list.push_back(tsr->values.data[lhs_idx]);
                for(std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++dim_idx){
                    merge_index_list[dim_idx].push_back(tsr->inds[dim_idx].data[lhs_idx]);
                }
                lhs_idx += 1;
            }else{
                merge_value_list.push_back(tsr->values.data[rhs_idx]);
                for(std::size_t dim_idx = 0; dim_idx < tsr->nmodes; ++rhs_idx){
                    merge_index_list[dim_idx].push_back(tsr->inds[dim_idx].data[rhs_idx]);
                }
                rhs_idx += 1;
            }
        }
        newSparseTensor(&tsr_add, merge_value_list.size(), ndims.data(), merge_value_list, merge_index_list);
        return tsr_add;
    }

//    Transpose2DBitfield::Transpose2DBitfield(std::size_t width) : m_width(width), m_bit_field(width*width){
//
//    }
//
//    bool Transpose2DBitfield::test_and_set(std::size_t row, std::size_t col) {
//        auto linear_index = row * m_width + col;
//        auto prev_value = m_bit_field.get(linear_index);
//        m_bit_field.set(linear_index);
//        return prev_value;
//    }
}
