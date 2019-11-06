#ifndef SISCI_NCCL_H_
#define SISCI_NCCL_H_

#include <nccl.h>

#define NCCLCHECK(call) do {                    \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

ncclDebugLogger_t ncclDebugLog;

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#define ncclCalloc(ptr, n)                          \
    ({                                              \
    ncclResult_t res;                               \
    size_t size = n*sizeof(**(ptr));                \
    *(ptr) = malloc(size);                          \
    if (*(ptr) == NULL) {                           \
        WARN("malloc failed");                      \
        res = ncclSystemError;                      \
    } else {                                        \
        memset(*(ptr), 0, size);                    \
        res = ncclSuccess;                          \
    }                                               \
    res;                                            \
    })

#endif
