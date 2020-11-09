/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "coordinates.hpp"
#include "runtime.hpp"
#include "task.hpp"
#include "tile.hpp"

#include "../../../../include/hip/hip_defines.h"

#include <cstdint>
#include <tuple>
#include <utility>

namespace hip
{
    namespace detail
    {
        template<typename F, typename... Args>
        inline
        void launch(
            const Dim3& num_blocks,
            const Dim3& dim_blocks,
            std::uint32_t group_mem_bytes,
            Stream* stream,
            F fn,
            std::tuple<Args...> args)
        {
            if (!stream) stream = Runtime::null_stream();

            stream->enqueue(Task{
                [=, fn = std::move(fn), args = std::move(args)](auto&&) {
                struct {
                    const decltype(fn)* fn_;
                    const std::tuple<Args...>* args_;

                    __HIP_TILE_FUNCTION__
                    void operator()() const noexcept
                    {
                        return std::apply(*fn_, *args_);
                    }
                } tmp{&fn, &args};

                const Tiled_domain domain{
                    dim_blocks, num_blocks, group_mem_bytes, std::move(tmp)};

                return for_each_tile(domain, fn, args);
            }});
        }
    } // Namespace hip::detail.
} // Namespace hip.