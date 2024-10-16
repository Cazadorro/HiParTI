//
// Created by User on 9/6/2024.
//
#include "hicoo_utils2.h"
namespace util {

    Transpose2DBitfield::Transpose2DBitfield(std::size_t width) : m_width(width), m_bit_field(width * width, false) {

    }

    bool Transpose2DBitfield::test_and_set(std::size_t row, std::size_t col) {
        auto linear_index = row * m_width + col;
        auto prev_value = m_bit_field.get(linear_index);
        m_bit_field.set(linear_index);
        return prev_value;
    }

    std::size_t Transpose2DBitfield::width() const {
        return m_width;
    }

    bool Transpose2DBitfield::get(std::size_t row, std::size_t col) const{
        auto linear_index = row * m_width + col;
        return m_bit_field.get(linear_index);
    }
}