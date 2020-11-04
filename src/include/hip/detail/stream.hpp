/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "task.hpp"

#include "../../../../external/moodycamel/blockingconcurrentqueue.h"

namespace hip
{
    namespace detail
    {
        using Stream = moodycamel::BlockingConcurrentQueue<Task>;
    } // Namespace hip::detail.
} // Namespace hip.