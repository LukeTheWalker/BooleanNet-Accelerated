#pragma once

#include <string>
#include <alpaka/alpaka.hpp>

namespace bit_ops {
    ALPAKA_FN_HOST_ACC constexpr auto set(auto& a, std::size_t b) noexcept {
        a |= (1ULL << b);
    }

    ALPAKA_FN_HOST_ACC constexpr auto clear(auto& a, std::size_t b) noexcept {
        a &= ~(1ULL << b);
    }

   ALPAKA_FN_HOST_ACC constexpr auto flip(auto& a, std::size_t b) noexcept {
        a ^= (1ULL << b);
    }

    ALPAKA_FN_HOST_ACC constexpr bool check(auto a, std::size_t b) noexcept {
        return (a & (1ULL << b)) != 0;
    }
}

ALPAKA_FN_HOST_ACC std::string get_impl_string(int impl);