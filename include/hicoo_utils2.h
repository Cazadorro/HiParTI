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
#include <iostream>
#include <sstream>
#include <csrk.h>
#include <unordered_map>
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
}


#endif //HIPARTI_HICOO_UTILS2_H
