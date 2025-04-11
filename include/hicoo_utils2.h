//
// Created by User on 9/6/2024.
//

#ifndef HIPARTI_HICOO_UTILS2_H
#define HIPARTI_HICOO_UTILS2_H
#include <czdr/bitutil/bit_vector.h>
#include <absl/container/flat_hash_map.h>
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../include/includes/sptensors.h"
#include "../src/sptensor/hicoo/hicoo.h"
#include "../include/includes/structs.h"
#include <csrk.h>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <ranges>
namespace util {


    //TODO could halv memory by doing upper/lower triangular.
    class Transpose2DBitfield {
    public:
        using word_type = std::uint64_t;
        static constexpr auto word_bit_count = std::numeric_limits<word_type>::digits;
        Transpose2DBitfield(std::size_t width);

        bool test_and_set(std::size_t row, std::size_t col);
        void reset(std::size_t row, std::size_t col);
        [[nodiscard]]
        bool get(std::size_t row, std::size_t col) const;
        [[nodiscard]]
        word_type get_word(std::size_t row, std::size_t col) const;

        [[nodiscard]]
        bool has_word(std::size_t row, std::size_t col) const;
        [[nodiscard]]
        std::size_t width() const;

        void clear();
        [[nodiscard]]
        bool is_identity() const;
        [[nodiscard]]
        bool is_empty() const;
        [[nodiscard]]
        std::size_t element_count() const;
        [[nodiscard]]
        std::size_t pop_count() const;
        [[nodiscard]]
        std::size_t bit_count() const;
        [[nodiscard]]
        std::size_t row_col_to_triangular_linear(std::size_t row, std::size_t col)const;
    private:

        std::size_t m_width = 0;
        std::size_t m_element_count = 0;
        // absl::flat_hash_map<std::size_t, word_type> m_word_map;
        std::unordered_map<std::size_t, word_type> m_word_map;
        // czdr::bit_vector<std::uint32_t> m_bit_field;

    };

    class CsrK {
    public:
        inline void clear() {
            row_ptrs.clear();
            col_ids.clear();
        }

        [[nodiscard]]
        inline std::size_t row_count() const{
            return (row_ptrs.size() -1);
        }
        [[nodiscard]]
        inline std::size_t nnz() const{
            return col_ids.size();
        }
        inline void inorder_append(std::uint32_t row, std::uint32_t col) {
            //should never happen.
            if ( (row + 1) < (row_ptrs.size()) ) {
                std::cout << fmt::format("row {} vs row_count {}\n", row, row_count()) << std::endl;
                std::abort();
            }
            if (row_ptrs.size() < (row + 1)) {
                //for all skipped rows until inserted one.
                for (std::size_t i = row_ptrs.size(); i < (row + 1); ++i) {
                    row_ptrs.push_back(col_ids.size());
                }
            }


            col_ids.push_back(col);
        }
        inline void append_last_row() {
            row_ptrs.push_back(col_ids.size());
        }

        inline void validate() {
            if (row_ptrs.back() != nnz()) {
                throw std::runtime_error("Row ptr max is not equal to nnz");
            }
            for (std::size_t i = 1; i < row_ptrs.size(); ++i) {
                if (!(row_ptrs[i] > row_ptrs[i - 1])) {
                    throw std::runtime_error("Row ptr unexpected difference");
                }
            }
            for (std::size_t i = 0; i < row_ptrs.size() - 1; ++i) {
                for (std::size_t j = row_ptrs[i]; j < row_ptrs[i + 1] - 1; ++j) {
                    if (!(col_ids[j] < col_ids[j+1])) {
                        throw std::runtime_error("Row ptr unexpected difference");
                    }
                }
            }
        }

        std::vector<std::uint32_t> row_ptrs;
        std::vector<std::uint32_t> col_ids;
    };


    class CsrKSymmetric {
    public:

        // static CsrK merge(const CsrK& lhs, const CsrK& rhs) {
        //     CsrK merged;
        //     auto max_row_size = std::max(lhs.row_ptrs.size(), rhs.row_ptrs.size());
        //     std::size_t lhs_idx = 0;
        //     std::size_t rhs_idx = 0;
        //     std::size_t lhs_col_id = 0;
        //     std::size_t rhs_col_id = 0;
        //     std::size_t last_lhs_segment = (lhs.row_ptrs.size() - 1);
        //     std::size_t last_rhs_segment = (rhs.row_ptrs.size() - 1);
        //     while ((lhs_idx < (lhs.row_ptrs.size() - 1)) || (rhs_idx < (rhs.row_ptrs.size() - 1))){
        //
        //         bool valid_lhs_segment = (lhs_idx < last_lhs_segment) && (lhs.row_ptrs[lhs_idx] != lhs.row_ptrs[lhs_idx + 1]);
        //         bool valid_rhs_segment = (rhs_idx < last_rhs_segment) && (lhs.row_ptrs[rhs_idx] != lhs.row_ptrs[rhs_idx + 1]);
        //
        //         if (!valid_lhs_segment) {
        //
        //         }
        //
        //         //if i is not at the last rle column segment, and segment not empty
        //         bool valid_lhs_segment = (i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1]);
        //         bool valid_rhs_segment = (i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1]);
        //         if ((i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1])) {
        //
        //         }
        //     }
        //     for (std::size_t i = 0; i <max_row_size -1; ++i) {
        //         std::size_t last_column_segment = (lhs.row_ptrs.size() - 1);
        //         //if i is not at the last rle column segment, and segment not empty
        //         bool valid_lhs_segment = (i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1]);
        //         bool valid_rhs_segment = (i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1]);
        //         if ((i < last_column_segment) && (lhs.row_ptrs[i] != lhs.row_ptrs[i + 1])) {
        //
        //         }
        //     }
        // }

        static CsrK create_symmetric(const CsrK& lhs, std::size_t diagonal_count) {
            CsrK merged;
            // double the number of non zeros, just not on the diagonal since they are at the same position transposed.
            merged.col_ids.resize(lhs.nnz() * 2 - diagonal_count);
            merged.row_ptrs.reserve(lhs.row_ptrs.size());

            //skip first one,
            for (auto row_idx = 0; row_idx <  lhs.row_ptrs.size() - 1; ++row_idx) {
                auto row_offset = lhs.row_ptrs[row_idx];
                for (auto col_idx = lhs.col_ids[row_offset]; col_idx < lhs.col_ids[row_offset + 1]; col_idx++) {
                    auto col_id = lhs.col_ids[col_idx];
                    if (row_idx >= merged.row_count()) {
                        merged.row_ptrs.push_back(0);
                    }
                    merged.row_ptrs[row_idx + 1] += 1;
                    if (col_id != row_idx) {
                        //transpose so using col_id as row id in this block.
                        if (col_id >= merged.row_count()) {
                            for (std::size_t i = merged.row_count(); i < col_id + 1; ++i) {
                                merged.row_ptrs.push_back(0);
                            }
                        }
                        merged.row_ptrs[col_id + 1] += 1;
                    }
                }
            }

            //accumulating offsets
            for (std::size_t i = 1; i < merged.row_ptrs.size(); i++) {
                merged.row_ptrs[i] = merged.row_ptrs[i] + merged.row_ptrs[i-1];
            }
            if (merged.row_ptrs.back() != merged.col_ids.size()) {

                std::abort();
            }

            for (auto row_idx = 0; row_idx <  lhs.row_ptrs.size() - 1; ++row_idx) {
                auto row_offset = lhs.row_ptrs[row_idx];
                auto new_row_offset = merged.row_ptrs[row_idx];
                auto col_offset = 0;
                auto old_offset_size = (lhs.row_ptrs[row_offset + 1] - lhs.row_ptrs[row_offset]);
                auto new_offset_size = (merged.row_ptrs[row_offset + 1] - merged.row_ptrs[row_offset]);
                //difference between entire row and how much to fill in from non transposed?
                if (old_offset_size > new_offset_size) {
                    std::abort();
                }
                auto transpose_offset = new_offset_size - old_offset_size;
                for (auto col_idx = lhs.col_ids[row_offset]; col_idx < lhs.col_ids[row_offset + 1]; col_idx++) {
                    auto col_id = lhs.col_ids[col_idx + 1];
                    //
                    merged.col_ids[new_row_offset + (transpose_offset + col_offset)] = col_id;
                    if (col_id != row_idx) {
                        auto transpose_new_row_offset = merged.row_ptrs[col_id];
                        merged.col_ids[transpose_new_row_offset + row_idx] = row_idx;
                    }
                    col_offset += 1;
                }
            }
            return merged;
        }

        void inorder_append_symmetric_element(std::uint32_t row, std::uint32_t col) {
            //TODO make start row at beginging and offset?
            // if (lower_triangular.row_count() == 0 && upper_triangular.row_count() == 0) {
            //     start_row = row;
            // }
            if (row == col) {
                diagonal_count += 1;
            }
            //ensures only upper triangular actually set.
            if (row > col) {
                //TODO big assumption was that was in order, but if row and col could be out of order, then this no longer makes sense?
                std::swap(row, col);
                if (lower_triangular.row_count() <= row) {
                    //for all skipped rows until inserted one.
                    for (std::size_t i = lower_triangular.row_count(); i < (row + 1); ++i) {
                        lower_triangular.row_ptrs.push_back(lower_triangular.col_ids.size());
                    }
                }
                lower_triangular.col_ids.push_back(col);
            }else {
                if (upper_triangular.row_count() <= row) {
                    //for all skipped rows until inserted one.
                    for (std::size_t i = upper_triangular.row_count(); i < (row + 1); ++i) {
                        upper_triangular.row_ptrs.push_back(upper_triangular.col_ids.size());
                    }
                }
                upper_triangular.col_ids.push_back(col);
            }

        }
        std::size_t start_row = 0;
        std::size_t diagonal_count = 0;
        CsrK upper_triangular;
        CsrK lower_triangular;
    };
}


#endif //HIPARTI_HICOO_UTILS2_H
