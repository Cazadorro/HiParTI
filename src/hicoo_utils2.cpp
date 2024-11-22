//
// Created by User on 9/6/2024.
//
#include "hicoo_utils2.h"
namespace util {

    Transpose2DBitfield::Transpose2DBitfield(std::size_t width) : m_width(width), m_bit_field(width * width, false) {

    }

    bool Transpose2DBitfield::test_and_set(std::size_t row, std::size_t col) {
        auto linear_index = row * m_width + col;
        auto transpose_linear_index = col * m_width + row;
        auto prev_value = m_bit_field.get(linear_index);
        m_bit_field.set(linear_index);
        m_bit_field.set(transpose_linear_index);
        return prev_value;
    }

    std::size_t Transpose2DBitfield::width() const {
        return m_width;
    }

    bool Transpose2DBitfield::get(std::size_t row, std::size_t col) const{
        auto linear_index = row * m_width + col;
        return m_bit_field.get(linear_index);
    }

    void Transpose2DBitfield::clear() {
        m_bit_field.fill(false);
    }

    bool Transpose2DBitfield::is_identity() const {
        for(std::size_t i = 0; i < m_width; ++i){
            auto row = i;
            auto col = i;
            auto linear_index = row * m_width + col;
            if(!m_bit_field.get(linear_index)){
                return false;
            }
        }
        if(element_count() == m_width){
            return true;
        }
    }

    bool Transpose2DBitfield::is_empty() const {
        return m_bit_field.empty();
    }

    std::size_t Transpose2DBitfield::element_count() const {
        return czdr::popcount(m_bit_field);
    }
}