# Frequently Asked Questions #

- [Who owns this project?](#who-owns-this-project)
- [Does it require an AMD CPU?](#does-it-require-an-amd-cpu)
- [Do I need the HIP-CPU-Compiler?](#do-i-need-the-hip-cpu-compiler)
- [Do I need ROCm?](#do-i-need-rocm)
- [Is it Linux-only?](#is-it-linux-only)
- [Can I interoperate with the HIP GPU Runtime?](#how-to-interoperate-with-the-hip-gpu-runtime)
- [Can I use it in C?](#can-I-use-it-in-c)
- [Can I use **_insert HIP library here_**?](#can-i-use-insert-hip-library-here)
- [I have a question and it is not answered here!](#i-have-a-question-and-it-is-not-answered-here)

## Who owns this project? ##

You, alongside every other past, present and future contributor:) Whilst initial
development was done at AMD, at the point when you are reading this everything
has been opened up. There is no closed-off, separate development, and there are
no special privileges for any particular group of contributors.

## Does it require an AMD CPU? ##

No, the HIP CPU Runtime is CPU agnostic. As long as you have a C++17 compliant
toolchain that can target it, the HIP CPU Runtime shall work with it.
[Please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
if you have identified a case where this does not hold, as it is a bug.

## Do I need the HIP-CPU-Compiler? ##

There is no such thing, the HIP CPU Runtime is compiler agnostic and does not
rely on any non-upstream / dedicated functionality. Otherwise stated, we expect
and want to compose with any C++17 compliant toolchain.
[Please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
if you have identified a case where this does not hold, as it is a bug.

## Do I need ROCm? ##

No, the HIP CPU Runtime does not depend on any [ROCm](https://github.com/RadeonOpenCompute/ROCm)
component. It is not necessary to install [ROCm](https://github.com/RadeonOpenCompute/ROCm)
in order to use the HIP CPU Runtime. Installing [ROCm](https://github.com/RadeonOpenCompute/ROCm)
shall not install the HIP CPU Runtime - this latter point might change at a some
point in the future, but there is no commitment to it at the moment.

## Is it Linux only? ##

No, the HIP CPU Runtime is OS agnostic. As long as a C++17 compliant toolchain
is available for some OS then the HIP CPU Runtime shall work in that
environment. [Please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
if you have identified a case where this does not hold, as it is a bug.

## Can I interoperate with the HIP GPU Runtime? ##

Not in an architected way. Since we expect the HIP CPU Runtime to undergo rapid,
iterative development, we decided to keep it as nimble as possible and not lock
into particularly restrictive decisions early on. As such, whilst convergence
with the HIP GPU Runtime is a valid medium-term goal, no cross-dependencies
exist at the moment. [Please create an issue](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
of you feel this is crucial for your use-case and is preventing experimentation
with the HIP CPU Runtime.

## Can I use it in C? ##

The HIP CPU Runtime is a C++ codebase, and cannot be used as-is from C. Whilst
it is possible and relatively straightforward to design and implement C bindings
for the HIP CPU Runtime public interface, this is not currently being pursued as
part of this project's development. External projects doing would be most
welcome.

## Can I use **_insert HIP library here_**? ##

Possibly - it depends on whether or not said library relies on any functionality
that is not implemented by the HIP CPU Runtime at the moment. Furthermore, not
all missing functionality is the same e.g. a missing element of the public
interface is likely to be added with alacrity, whereas inline GCN ASM shall
never work. Thus, whilst a library relying on the former is highly likely to be
usable quite rapidly, a library relying on the latter shall never work.

## I have a question and it is not answered here! ##

Please help with filling up this gap:

- [create an issue about your question](https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/new/choose)
- or, even better, [create a pull request with the question and a proposed answer](https://github.com/ROCm-Developer-Tools/HIP-CPU/pulls)

Whichever option you choose, thank you for helping!