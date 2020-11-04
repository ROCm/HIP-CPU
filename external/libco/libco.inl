#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wparentheses"

  /* placing code in section(text) does not mark it executable with Clang. */
  #undef  LIBCO_MPROTECT
  #define LIBCO_MPROTECT
#endif

#if defined(__clang__) || defined(__GNUC__)
  #if defined(__i386__)
    #include "x86.inl"
  #elif defined(__amd64__)
    #include "amd64.inl"
  #elif defined(__arm__)
    #include "arm.inl"
  #elif defined(__aarch64__)
    #include "aarch64.inl"
  #elif defined(__powerpc64__) && defined(_CALL_ELF) && _CALL_ELF == 2
    #include "ppc64v2.inl"
  #elif defined(_ARCH_PPC) && !defined(__LITTLE_ENDIAN__)
    #include "ppc.inl"
  #elif defined(_WIN32)
    #include "fiber.inl"
  #else
    #include "sjlj.inl"
  #endif
#elif defined(_MSC_VER)
  #if defined(_M_IX86)
    #include "x86.inl"
  #elif defined(_M_AMD64)
    #include "amd64.inl"
  #else
    #include "fiber.inl"
  #endif
#else
  #error "libco: unsupported processor, compiler or operating system"
#endif
