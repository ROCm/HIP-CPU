/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "flat_combiner.hpp"
#include "task.hpp"

#include <vector>

namespace hip
{
    namespace detail
    {
        using Stream = Flat_combiner<std::vector<Task>>;
    } // Namespace hip::detail.
} // Namespace hip.