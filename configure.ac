AC_INIT([sisci-nccl], [0.1], [pci-support@dolphinics.com])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AM_PROG_AR

AC_PROG_CC
AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

AC_ARG_WITH([dis],
            AC_HELP_STRING([--with-dis=PATH], [Path to DIS installation]),
            [DIS_HOME="$withval"],
            [DIS_HOME="/opt/DIS"])

AS_IF([test -d $DIS_HOME/lib64], [dis_libdir="lib64"], [dis_libdir="lib"])
CFLAGS="-I$DIS_HOME/include -I$DIS_HOME/include/dis $CFLAGS"
LDFLAGS="-L$DIS_HOME/$dis_libdir $LDFLAGS"

AC_ARG_WITH([cuda],
            AC_HELP_STRING([--with-dis=PATH], [Path to CUDA installation]),
            [CUDA_HOME="$withval"],
            [CUDA_HOME="/usr/local/cuda"])

AS_IF([test -d $CUDA_HOME/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
CFLAGS="-I$CUDA_HOME/include $CFLAGS"
LDFLAGS="-L$CUDA_HOME/$cuda_libdir $LDFLAGS"

AC_CHECK_HEADERS([limits.h stdlib.h string.h unistd.h sisci_api.h cuda_runtime_api.h nccl.h], [],[
	AC_MSG_ERROR([unable to find required headers])
])

AC_SEARCH_LIBS([_SISCI_PUBLIC_FUNC_ST_SCIInitialize], [sisci], [], [
  AC_MSG_ERROR([unable to find libsisci])
])

AC_SEARCH_LIBS([cudaSetDevice], [cudart], [], [
  AC_MSG_ERROR([unable to find libcudart])
])

AC_SEARCH_LIBS([ncclCommInitRank], [nccl], [], [
  AC_MSG_ERROR([unable to find libnccl])
])

LT_INIT([shared disable-static])

CFLAGS="-std=gnu11 -Wall -Wextra $CFLAGS"

AC_PROG_LIBTOOL

AC_OUTPUT
