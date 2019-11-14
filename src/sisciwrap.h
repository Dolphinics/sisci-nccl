#ifndef SISCIWRAP_H_
#define SISCIWRAP_H_

#include <sisci_api.h>
#include <sisci_error.h>
#include <sisci_types.h>
#include <nccl.h>
#include <nccl_net.h>

#include "sisci_nccl.h"

#define SISCI_CHECK(function, error, quiet)                             \
    ({                                                                  \
        ncclResult_t result = ncclSuccess;                              \
        if (error != SCI_ERR_OK) {                                      \
            if (!quiet) {                                               \
                WARN("%s: %s", function,                                \
                     SCIGetErrorString(error));                         \
            }                                                           \
            result = ncclInternalError;                                 \
        }                                                               \
        result;                                                         \
    })                                                                  \


#define SISCI_WRAP(fn, ...)                     \
    ({                                          \
        sci_error_t err;                        \
        fn(__VA_ARGS__, &err);                  \
        SISCI_CHECK(#fn, err, 0);               \
    })

#define SISCI_WRAP_RET(fn, ret, ...)                        \
    ({                                                      \
        sci_error_t err;                                    \
        *(ret) = fn(__VA_ARGS__, &err);                     \
        SISCI_CHECK(#fn, err, 1);                           \
    })

#define SISCI_WRAP_NOERROR(fn, ret, ...)                        \
    ({                                                          \
        *(ret) = fn(__VA_ARGS__);                               \
        ncclSuccess;                                            \
    })

#define ncclSCIInitialize(...)                         SISCI_WRAP(SCIInitialize, __VA_ARGS__)
#define ncclSCITerminate(...)                          SISCI_WRAP(SCITerminate, __VA_ARGS__)
#define ncclSCIOpen(...)                               SISCI_WRAP(SCIOpen, __VA_ARGS__)
#define ncclSCIClose(...)                              SISCI_WRAP(SCIClose, __VA_ARGS__)
#define ncclSCIConnectSegment(...)                     SISCI_WRAP(SCIConnectSegment, __VA_ARGS__)
#define ncclSCIDisconnectSegment(...)                  SISCI_WRAP(SCIDisconnectSegment, __VA_ARGS__)
#define ncclSCIGetRemoteSegmentSize(...)               SISCI_WRAP(SCIGetRemoteSegmentSize, __VA_ARGS__)
#define ncclSCIGetRemoteSegmentId(...)                 SISCI_WRAP(SCIGetRemoteSegmentId, __VA_ARGS__)
#define ncclSCIGetRemoteSegmentNodeId(...)             SISCI_WRAP(SCIGetRemoteSegmentNodeId, __VA_ARGS__)
#define ncclSCIWaitForRemoteSegmentEvent(...)          SISCI_WRAP(SCIWaitForRemoteSegmentEvent, __VA_ARGS__)
#define ncclSCIMapRemoteSegment(ret, ...)              SISCI_WRAP_RET(SCIMapRemoteSegment, ret, __VA_ARGS__)
#define ncclSCIMapLocalSegment(ret, ...)               SISCI_WRAP_RET(SCIMapLocalSegment, ret,  __VA_ARGS__)
#define ncclSCIUnmapSegment(...)                       SISCI_WRAP(SCIUnmapSegment, __VA_ARGS__)
#define ncclSCICreateSegment(...)                      SISCI_WRAP(SCICreateSegment, __VA_ARGS__)
#define ncclSCIWaitForLocalSegmentEvent(...)           SISCI_WRAP(SCIWaitForLocalSegmentEvent, __VA_ARGS__)
#define ncclSCIPrepareSegment(...)                     SISCI_WRAP(SCIPrepareSegment, __VA_ARGS__)
#define ncclSCIRemoveSegment(...)                      SISCI_WRAP(SCIRemoveSegment, __VA_ARGS__)
#define ncclSCIGetLocalSegmentSize(...)                SISCI_WRAP(SCIGetLocalSegmentSize, __VA_ARGS__)
#define ncclSCIGetLocalSegmentId(...)                  SISCI_WRAP(SCIGetLocalSegmentId, __VA_ARGS__)
#define ncclSCISetSegmentAvailable(...)                SISCI_WRAP(SCISetSegmentAvailable, __VA_ARGS__)
#define ncclSCISetSegmentUnavailable(...)              SISCI_WRAP(SCISetSegmentUnavailable, __VA_ARGS__)
#define ncclSCICreateMapSequence(...)                  SISCI_WRAP(SCICreateMapSequence, __VA_ARGS__)
#define ncclSCIRemoveSequence(...)                     SISCI_WRAP(SCIRemoveSequence, __VA_ARGS__)
#define ncclSCIStartSequence(...)                      SISCI_WRAP(SCIStartSequence, __VA_ARGS__)
#define ncclSCICheckSequence(...)                      SISCI_WRAP(SCICheckSequence, __VA_ARGS__)
#define ncclSCIStoreBarrier(...)                       SISCI_WRAP(SCIStoreBarrier, __VA_ARGS__)
#define ncclSCIProbeNode(...)                          SISCI_WRAP(SCIProbeNode, __VA_ARGS__)
#define ncclSCIGetCSRRegister(...)                     SISCI_WRAP(SCIGetCSRRegister, __VA_ARGS__)
#define ncclSCISetCSRRegister(...)                     SISCI_WRAP(SCISetCSRRegister, __VA_ARGS__)
#define ncclSCIGetLocalCSR(...)                        SISCI_WRAP(SCIGetLocalCSR, __VA_ARGS__)
#define ncclSCISetLocalCSR(...)                        SISCI_WRAP(SCISetLocalCSR, __VA_ARGS__)
#define ncclSCIAttachPhysicalMemory(...)               SISCI_WRAP(SCIAttachPhysicalMemory, __VA_ARGS__)
#define ncclSCIQuery(...)                              SISCI_WRAP(SCIQuery, __VA_ARGS__)
#define ncclSCIGetLocalNodeId(...)                     SISCI_WRAP(SCIGetLocalNodeId, __VA_ARGS__)
#define ncclSCIGetNodeIdByAdapterName(...)             SISCI_WRAP(SCIGetNodeIdByAdapterName, __VA_ARGS__)
#define ncclSCIGetNodeInfoByAdapterName(...)           SISCI_WRAP(SCIGetNodeInfoByAdapterName, __VA_ARGS__)
#define ncclSCIGetErrorString(...)                     SISCI_WRAP(SCIGetErrorString, __VA_ARGS__)
#define ncclSCIGetAdapterTypeString(...)               SISCI_WRAP(SCIGetAdapterTypeString, __VA_ARGS__)
#define ncclSCIGetAdapterFamilyString(...)             SISCI_WRAP(SCIGetAdapterFamilyString, __VA_ARGS__)
#define ncclSCICreateDMAQueue(...)                     SISCI_WRAP(SCICreateDMAQueue, __VA_ARGS__)
#define ncclSCIRemoveDMAQueue(...)                     SISCI_WRAP(SCIRemoveDMAQueue, __VA_ARGS__)
#define ncclSCIAbortDMAQueue(...)                      SISCI_WRAP(SCIAbortDMAQueue, __VA_ARGS__)
#define ncclSCIDMAQueueState(ret, ...)                 SISCI_WRAP_NOERROR(SCIDMAQueueState, ret, __VA_ARGS__)
#define ncclSCIWaitForDMAQueue(...)                    SISCI_WRAP(SCIWaitForDMAQueue, __VA_ARGS__)
#define ncclSCICreateNamedInterrupt(...)               SISCI_WRAP(SCICreateNamedInterrupt, __VA_ARGS__)
#define ncclSCICreateInterrupt(...)                    SISCI_WRAP(SCICreateInterrupt, __VA_ARGS__)
#define ncclSCIRemoveInterrupt(...)                    SISCI_WRAP(SCIRemoveInterrupt, __VA_ARGS__)
#define ncclSCIWaitForInterrupt(...)                   SISCI_WRAP(SCIWaitForInterrupt, __VA_ARGS__)
#define ncclSCIConnectInterrupt(...)                   SISCI_WRAP(SCIConnectInterrupt, __VA_ARGS__)
#define ncclSCIDisconnectInterrupt(...)                SISCI_WRAP(SCIDisconnectInterrupt, __VA_ARGS__)
#define ncclSCITriggerInterrupt(...)                   SISCI_WRAP(SCITriggerInterrupt, __VA_ARGS__)
#define ncclSCICreateDataInterrupt(...)                SISCI_WRAP(SCICreateDataInterrupt, __VA_ARGS__)
#define ncclSCIRemoveDataInterrupt(...)                SISCI_WRAP(SCIRemoveDataInterrupt, __VA_ARGS__)
#define ncclSCIWaitForDataInterrupt(...)               SISCI_WRAP(SCIWaitForDataInterrupt, __VA_ARGS__)
#define ncclSCIConnectDataInterrupt(...)               SISCI_WRAP(SCIConnectDataInterrupt, __VA_ARGS__)
#define ncclSCIDisconnectDataInterrupt(...)            SISCI_WRAP(SCIDisconnectDataInterrupt, __VA_ARGS__)
#define ncclSCITriggerDataInterrupt(...)               SISCI_WRAP(SCITriggerDataInterrupt, __VA_ARGS__)
#define ncclSCIRegisterInterruptFlag(...)              SISCI_WRAP(SCIRegisterInterruptFlag, __VA_ARGS__)
#define ncclSCIEnableConditionalInterrupt(...)         SISCI_WRAP(SCIEnableConditionalInterrupt, __VA_ARGS__)
#define ncclSCIDisableConditionalInterrupt(...)        SISCI_WRAP(SCIDisableConditionalInterrupt, __VA_ARGS__)
#define ncclSCIGetConditionalInterruptTrigCounter(...) SISCI_WRAP(SCIGetConditionalInterruptTrigCounter, __VA_ARGS__)
#define ncclSCIMemWrite(...)                           SISCI_WRAP(SCIMemWrite, __VA_ARGS__)
#define ncclSCIMemCpy(...)                             SISCI_WRAP(SCIMemCpy, __VA_ARGS__)
#define ncclSCIMemCpy_dual(...)                        SISCI_WRAP(SCIMemCpy_dual, __VA_ARGS__)
#define ncclSCIRegisterSegmentMemory(...)              SISCI_WRAP(SCIRegisterSegmentMemory, __VA_ARGS__)
#define ncclSCIConnectSCISpace(...)                    SISCI_WRAP(SCIConnectSCISpace, __VA_ARGS__)
#define ncclSCIAttachLocalSegment(...)                 SISCI_WRAP(SCIAttachLocalSegment, __VA_ARGS__)
#define ncclSCIShareSegment(...)                       SISCI_WRAP(SCIShareSegment, __VA_ARGS__)
#define ncclSCIFlush(...)                              SISCI_WRAP(SCIFlush, __VA_ARGS__)
#define ncclSCIStartDmaTransfer(...)                   SISCI_WRAP(SCIStartDmaTransfer, __VA_ARGS__)
#define ncclSCIStartDmaTransferMem(...)                SISCI_WRAP(SCIStartDmaTransferMem, __VA_ARGS__)
#define ncclSCIStartDmaTransferVec(...)                SISCI_WRAP(SCIStartDmaTransferVec, __VA_ARGS__)
#define ncclDISStartDmaTransfer(...)                   SISCI_WRAP(DISStartDmaTransfer, __VA_ARGS__)
#define ncclDISStartDmaTransferVec(...)                SISCI_WRAP(DISStartDmaTransferVec, __VA_ARGS__)
#define ncclSCIRequestDMAChannel(...)                  SISCI_WRAP(SCIRequestDMAChannel, __VA_ARGS__)
#define ncclSCIReturnDMAChannel(...)                   SISCI_WRAP(SCIReturnDMAChannel, __VA_ARGS__)
#define ncclSCIAssignDMAChannel(...)                   SISCI_WRAP(SCIAssignDMAChannel, __VA_ARGS__)
#define ncclGetSciMemCopyFunction(...)                 SISCI_WRAP(GetSciMemCopyFunction, __VA_ARGS__)
#define ncclSetSciMemCopyFunction(...)                 SISCI_WRAP(SetSciMemCopyFunction, __VA_ARGS__)
#define ncclSCICacheSync(...)                          SISCI_WRAP(SCICacheSync, __VA_ARGS__)
#define ncclSCIRegisterPCIeRequester(...)              SISCI_WRAP(SCIRegisterPCIeRequester, __VA_ARGS__)
#define ncclSCIUnregisterPCIeRequester(...)            SISCI_WRAP(SCIUnregisterPCIeRequester, __VA_ARGS__)
#define ncclSCIBorrowDevice(...)                       SISCI_WRAP(SCIBorrowDevice, __VA_ARGS__)
#define ncclSCIReturnDevice(...)                       SISCI_WRAP(SCIReturnDevice, __VA_ARGS__)
#define ncclSCIReleaseExclusiveBorrow(...)             SISCI_WRAP(SCIReleaseExclusiveBorrow, __VA_ARGS__)
#define ncclSCIConnectDeviceSegment(...)               SISCI_WRAP(SCIConnectDeviceSegment, __VA_ARGS__)
#define ncclSCIConnectDeviceSegmentPath(...)           SISCI_WRAP(SCIConnectDeviceSegmentPath, __VA_ARGS__)
#define ncclSCICreateDeviceSegment(...)                SISCI_WRAP(SCICreateDeviceSegment, __VA_ARGS__)
#define ncclSCIMapLocalSegmentForDevice(...)           SISCI_WRAP(SCIMapLocalSegmentForDevice, __VA_ARGS__)
#define ncclSCIUnmapLocalSegmentForDevice(...)         SISCI_WRAP(SCIUnmapLocalSegmentForDevice, __VA_ARGS__)
#define ncclSCIMapRemoteSegmentForDevice(...)          SISCI_WRAP(SCIMapRemoteSegmentForDevice, __VA_ARGS__)
#define ncclSCIUnmapRemoteSegmentForDevice(...)        SISCI_WRAP(SCIUnmapRemoteSegmentForDevice, __VA_ARGS__)
#define ncclSCIGetDeviceList(...)                      SISCI_WRAP(SCIGetDeviceList, __VA_ARGS__)
#define ncclSCIGetFabricDeviceId(...)                  SISCI_WRAP(SCIGetFabricDeviceId, __VA_ARGS__)

#endif //End include guard
