//
// Created by User on 9/6/2024.
//
#include "hicoo_utils2.h"
#include <utility>
namespace util {

    Transpose2DBitfield::Transpose2DBitfield(std::size_t width) : m_width(width){//, m_bit_field(width * width, false) {
            // m_word_map.reserve((width * (width + 1) / 2) / word_bit_count);
        m_word_map.reserve((width * width)/ word_bit_count);
    }

    std::size_t Transpose2DBitfield::row_col_to_triangular_linear(std::size_t row, std::size_t col) const{
        //return row*(row+1)/2 + col;
        return row * m_width + col;
    }
    bool Transpose2DBitfield::test_and_set(std::size_t row, std::size_t col) {
        auto linear_bit_index = row_col_to_triangular_linear(row, col);
        auto linear_word_index = linear_bit_index / word_bit_count;
        auto transpose_bit_index  = row_col_to_triangular_linear(col, row);
        auto transpose_word_index = transpose_bit_index / word_bit_count;

        // std::cout << m_word_map.size() << std::endl;
        // assert(m_word_map.size() > linear_word_index);
        auto [itr, ignored_0] = m_word_map.try_emplace(linear_word_index, 0u);

        auto word_bit_index = linear_bit_index % word_bit_count;
        auto transpose_word_bit_index = transpose_bit_index % word_bit_count;
        if (itr == m_word_map.end()) {
            std::cout << "huh?" << std::endl;
            std::abort();
        }
        auto prev_value = (itr->second & (static_cast<word_type>(1u) << word_bit_index)) != 0;
        if (!prev_value) {
            itr->second |= (static_cast<word_type>(1u) << word_bit_index);
            if (row == col) {
                m_element_count += 1;
            }else {
                auto [itr_t, ignored_1] = m_word_map.try_emplace(transpose_word_index, 0u);
                itr_t->second |= (static_cast<word_type>(1u) << transpose_word_bit_index);
                m_element_count += 2;
            }
        }
        return prev_value;
    }

    void Transpose2DBitfield::reset(std::size_t row, std::size_t col) {
        auto linear_bit_index = row_col_to_triangular_linear(row, col);
        auto linear_word_index = linear_bit_index / word_bit_count;
        auto transpose_bit_index  = row_col_to_triangular_linear(col, row);
        auto transpose_word_index = transpose_bit_index / word_bit_count;


        if (auto itr = m_word_map.find(linear_word_index); itr != m_word_map.end()) {
            auto word_bit_index = linear_bit_index % word_bit_count;
            auto transpose_word_bit_index = transpose_bit_index % word_bit_count;
            if (itr->second & (static_cast<word_type>(1u) << word_bit_index) != 0) {
                auto itr_t = m_word_map.find(transpose_word_index);
                itr->second &= ~(static_cast<word_type>(1u) << word_bit_index);

                if (row == col) {
                    m_element_count -= 1;
                }else {
                    itr_t-> second &= ~(static_cast<word_type>(1u) << (transpose_word_bit_index));
                    m_element_count -= 2;
                }
            }
        }
    }

    std::size_t Transpose2DBitfield::width() const {
        return m_width;
    }

    Transpose2DBitfield::word_type Transpose2DBitfield::get_word(std::size_t row, std::size_t col) const {
        auto linear_bit_index = row_col_to_triangular_linear(row, col);
        auto linear_word_index = linear_bit_index / word_bit_count;
        auto transpose_bit_index  = row_col_to_triangular_linear(col, row);
        auto transpose_word_index = transpose_bit_index / word_bit_count;

        if (auto itr = m_word_map.find(linear_word_index); itr != m_word_map.end()) {
            return itr->second;
        }
        return 0;
    }

    bool Transpose2DBitfield::has_word(std::size_t row, std::size_t col) const {
        auto linear_bit_index = row_col_to_triangular_linear(row, col);
        auto linear_word_index = linear_bit_index / word_bit_count;
        auto transpose_bit_index  = row_col_to_triangular_linear(col, row);
        auto transpose_word_index = transpose_bit_index / word_bit_count;
        return  m_word_map.contains(linear_word_index);
    }

    bool Transpose2DBitfield::get(std::size_t row, std::size_t col) const{
        auto linear_bit_index = row_col_to_triangular_linear(row, col);
        auto linear_word_index = linear_bit_index / word_bit_count;
        auto word_bit_index = linear_bit_index % word_bit_count;
        auto transpose_bit_index  = row_col_to_triangular_linear(col, row);
        auto transpose_word_index = transpose_bit_index / word_bit_count;
        return (get_word(row, col) & (static_cast<word_type>(1u) << word_bit_index)) != 0;
    }



    void Transpose2DBitfield::clear() {
        m_word_map.clear();
        m_element_count = 0;
    }

    bool Transpose2DBitfield::is_identity() const {\
        if(element_count() != m_width){
            return false;
        }
        for(std::size_t i = 0; i < m_width; ++i){
            auto row = i;
            auto col = i;
            if(!get(row, col)){
                return false;
            }
        }
        return true;
    }

    bool Transpose2DBitfield::is_empty() const {
        return m_element_count == 0;
    }

    std::size_t Transpose2DBitfield::element_count() const {
        return m_element_count;
    }

    std::size_t Transpose2DBitfield::pop_count() const {
        std::size_t sum = 0;
        for (auto value: m_word_map | views::values) {
            sum += std::popcount(value);
        }
        return sum;
    }

    std::size_t Transpose2DBitfield::bit_count() const {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < m_width; ++i) {
            for (std::size_t j = 0; j < m_width; ++j) {
                sum+= get(i,j) ? 1 : 0;
            }
        }
        return sum;
    }
}
