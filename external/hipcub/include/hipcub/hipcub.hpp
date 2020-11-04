#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <iterator>
#include <numeric>
#include <limits>
#include <type_traits>

namespace hipcub
{
    namespace DeviceHistogram
    {
        template<typename I, typename H, typename L, typename DX>
        inline
        hipError_t HistogramEven(
            void* d_temp_storage,
            std::size_t& temp_storage_bytes,
            I d_samples,
            H* d_histogram,
            int num_levels,
            L lower_level,
            L upper_level,
            DX num_samples,
            hipStream_t /*stream*/ = nullptr,
            bool /*debug_synchronous*/ = false)
        {
            if (!d_temp_storage) {
                temp_storage_bytes = 1; // TODO: placeholder, not used.

                return hipSuccess;
            }

            if (num_samples == 0) return hipSuccess;
            if (num_levels < 2) return hipErrorInvalidValue;

            const auto den{upper_level - lower_level};
            std::for_each_n(d_samples, num_samples, [=](auto&& x) {
                const auto bin{static_cast<unsigned int>(std::clamp(
                    num_levels * (x - lower_level) / den,
                    L{0},
                    L(num_levels - 1)))};
                ++d_histogram[bin];
            });

            return hipSuccess;
        }

        template<
            int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename I,
            typename H,
            typename L,
            typename DX>
        inline
        hipError_t MultiHistogramEven(
            void* d_temp_storage,
            std::size_t& temp_storage_bytes,
            I d_samples,
            H* d_histogram[NUM_ACTIVE_CHANNELS],
            int num_levels[NUM_ACTIVE_CHANNELS],
            L lower_level[NUM_ACTIVE_CHANNELS],
            L upper_level[NUM_ACTIVE_CHANNELS],
            DX num_pixels,
            hipStream_t /*stream*/ = nullptr,
            bool /*debug_synchronous*/ = false)
        {
            if (!d_temp_storage) {
                temp_storage_bytes = 1; // TODO: placeholder, not used.

                return hipSuccess;
            }

            const auto valid{std::none_of(
                num_levels,
                num_levels + NUM_ACTIVE_CHANNELS,
                [](auto&& x) { return x < 2; })};

            if (num_pixels == 0) return hipSuccess;
            if (!valid) return hipErrorInvalidValue;

            using T = typename std::iterator_traits<I>::value_type;
            using A = std::array<T, NUM_CHANNELS>;

            const auto den{[=]() {
                std::array<L, NUM_ACTIVE_CHANNELS> r;
                for (auto i = 0u; i != NUM_ACTIVE_CHANNELS; ++i) {
                    r[i] = upper_level[i] - lower_level[i];
                }

                return r;
            }()};

            std::for_each_n(
                reinterpret_cast<const A*>(&*d_samples),
                num_pixels / NUM_CHANNELS,
                [=, &den](auto&& x) {
                for (auto i = 0u; i != NUM_ACTIVE_CHANNELS; ++i) {
                    const auto bin{static_cast<unsigned int>(std::clamp(
                        num_levels[i] * (x[i] - lower_level[i]) / den[i],
                        L{0},
                        L(num_levels[i] - 1)))};
                        ++d_histogram[i][bin];
                }
            });

            return hipSuccess;
        }
    } // Namespace hipcub::DeviceHistogram.

    namespace DeviceReduce
    {
        template<typename I, typename O>
        inline
        hipError_t ArgMax(
            void* d_temp_storage,
            std::size_t& temp_storage_bytes,
            I d_in,
            O d_out,
            int num_items,
            hipStream_t /*stream*/ = nullptr,
            bool /*debug_synchronous*/ = false)
        {
            if (!d_temp_storage) {
                temp_storage_bytes = 1; // TODO: placeholder, not used.

                return hipSuccess;
            }

            using T = std::decay_t<decltype(d_out->value)>;

            if (num_items == 0) {
                *d_out = {1, std::numeric_limits<T>::lowest()};

                return hipSuccess;
            }

            const auto it{std::max_element(
                std::execution::par_unseq, d_in, d_in + num_items)};
            d_out->key = std::distance(d_in, it);
            d_out->value = *it;

            return hipSuccess;
        }
    } // Namespace hipcub::DeviceReduce.

    namespace DeviceScan
    {
        template<typename I, typename O>
        inline
        hipError_t ExclusiveSum(
            void* d_temp_storage,
            std::size_t& temp_storage_bytes,
            I d_in,
            O d_out,
            std::int32_t num_items,
            hipStream_t /*stream*/ = nullptr,
            bool /*debug_synchronous*/ = false)
        {
            if (!d_temp_storage) {
                temp_storage_bytes = 1; // TODO: placeholder, not used.

                return hipSuccess;
            }

            if (num_items == 0) return hipSuccess;

            using T = typename std::iterator_traits<O>::value_type;

            std::exclusive_scan(
                std::execution::par_unseq, d_in, d_in + num_items, d_out, T{0});

            return hipSuccess;
        }
    } // Namespace hipcub::DeviceScan.
} // Namespace hipcub.