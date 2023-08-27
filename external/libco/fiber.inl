#define LIBCO_C
#include "libco.h"
#include "settings.h"

#define WINVER 0x0400
#define _WIN32_WINNT 0x0400
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

inline thread_local cothread_t co_active_ = 0;

inline void __stdcall co_thunk(void* coentry) {
  ((void (*)(void))coentry)();
}

inline cothread_t co_active(void) {
  if(!co_active_) {
    ConvertThreadToFiber(0);
    co_active_ = GetCurrentFiber();
  }
  return co_active_;
}

inline cothread_t co_derive(void* memory, unsigned int heapsize, void (*coentry)(void)) {
  /* Windows fibers do not allow users to supply their own memory */
  return (cothread_t)0;
}

inline cothread_t co_create(unsigned int heapsize, void (*coentry)(void)) {
  if(!co_active_) {
    ConvertThreadToFiber(0);
    co_active_ = GetCurrentFiber();
  }
  return (cothread_t)CreateFiber(heapsize, co_thunk, (void*)coentry);
}

inline void co_delete(cothread_t cothread) {
  DeleteFiber(cothread);
}

inline void co_switch(cothread_t cothread) {
  co_active_ = cothread;
  SwitchToFiber(cothread);
}

inline int co_serializable(void) {
  return 0;
}

#ifdef __cplusplus
}
#endif
