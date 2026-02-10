#pragma once

#include <string>
#include <cstdint>

namespace bit_ops {
    constexpr void set(uint64_t& a, std::size_t b) noexcept {
        a |= (1ULL << b);
    }

    constexpr void clear(uint64_t& a, std::size_t b) noexcept {
        a &= ~(1ULL << b);
    }

   constexpr void flip(uint64_t& a, std::size_t b) noexcept {
        a ^= (1ULL << b);
    }

    constexpr bool check(uint64_t a, std::size_t b) noexcept {
        return (a & (1ULL << b)) != 0;
    }
}

std::string get_impl_string(int impl);