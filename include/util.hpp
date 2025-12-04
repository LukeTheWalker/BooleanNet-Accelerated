#pragma once

#include <string>
#include <alpaka/alpaka.hpp>

namespace bit_ops {
    ALPAKA_FN_HOST_ACC constexpr void set(uint64_t& a, std::size_t b) noexcept {
        a |= (1ULL << b);
    }

    ALPAKA_FN_HOST_ACC constexpr void clear(uint64_t& a, std::size_t b) noexcept {
        a &= ~(1ULL << b);
    }

   ALPAKA_FN_HOST_ACC constexpr void flip(uint64_t& a, std::size_t b) noexcept {
        a ^= (1ULL << b);
    }

    ALPAKA_FN_HOST_ACC constexpr bool check(uint64_t a, std::size_t b) noexcept {
        return (a & (1ULL << b)) != 0;
    }
}

ALPAKA_FN_HOST_ACC std::string get_impl_string(int impl);