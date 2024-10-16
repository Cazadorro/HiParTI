//
// Created by User on 9/6/2024.
//

#ifndef HIPARTI_HICOO_UTILS2_H
#define HIPARTI_HICOO_UTILS2_H
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
    class Transpose2DBitfield {
    public:
        Transpose2DBitfield(std::size_t width);

        bool test_and_set(std::size_t row, std::size_t col);

        [[nodiscard]]
        bool get(std::size_t row, std::size_t col) const;
        [[nodiscard]]
        std::size_t width() const;

        void clear();
    private:
        std::size_t m_width;
        czdr::bit_vector<std::uint32_t> m_bit_field;

    };
}


#endif //HIPARTI_HICOO_UTILS2_H
