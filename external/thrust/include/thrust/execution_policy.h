#pragma once

#include <execution>

namespace thrust
{
    namespace hip
    {
        using std::execution::par;
        using std::execution::par_unseq;
        using std::execution::seq;
    } // Namespace thrust::hip.
} // Namespace thrust.