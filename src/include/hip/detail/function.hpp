/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "module.hpp"

#if defined(_WIN32)
    #if !defined(NOMINMAX)
        #define NOMINMAX
    #endif
    #if !defined(WIN32_LEAN_AND_MEAN)
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include <string_view>

namespace hip
{
    namespace detail
    {
        // BEGIN CLASS FUNCTION
        class Function final {
            // DATA
            void (*fn_)();
            const Module* m_;

            // FRIENDS - ACCESSORS
            friend
            inline
            decltype(auto) native_handle(const Function& x) noexcept
            {
                return x.native_handle();
            }
            friend
            inline
            const Module& parent(const Function& x) noexcept
            {
                return x.parent();
            }
            friend
            inline
            bool valid(const Function& x) noexcept
            {
                return x.valid();
            }
        public:
            using native_handle_type = decltype(fn_);

            // CREATORS
            Function() = default;
            Function(const Module& m, std::string_view fn) noexcept;
            Function(const Function&) = default;
            Function(Function&&) = default;
            ~Function() = default;

            // MANIPULATORS
            Function& operator=(const Function&) = default;
            Function& operator=(Function&&) = default;

            // ACCESSORS
            native_handle_type native_handle() const noexcept;
            const Module& parent() const noexcept;
            bool valid() const noexcept;
        };

        // CREATORS
        inline
        Function::Function(const Module& m, std::string_view fn) noexcept
            :
            fn_{reinterpret_cast<native_handle_type>(address_of(m, fn))}, m_{&m}
        {}

        // ACCESSORS
        inline
        typename Function::native_handle_type
            Function::native_handle() const noexcept
        {
            return fn_;
        }

        inline
        const Module& Function::parent() const noexcept
        {
            return *m_;
        }

        inline
        bool Function::valid() const noexcept
        {
            return fn_;
        }
        // END CLASS FUNCTION
    } // Namespace hip::detail.
} // Namespace hip.