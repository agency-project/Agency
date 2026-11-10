"""Seccomp filter construction for agproxy_ptrace, via the `pyseccomp`
library (pure-Python ctypes bindings to libseccomp, API-compatible with
libseccomp's own Python bindings).

The filter's default action is ALLOW -- agproxy_ptrace does not sandbox the
traced process by denying syscalls wholesale, it only asks the kernel to
stop-and-report the specific syscalls the caller wants observed
(`SECCOMP_RET_TRACE`, i.e. pyseccomp's `TRACE` action). Every other syscall
runs untouched, at native speed, with no ptrace-stop overhead at all --
that's the whole point of installing a filter instead of full
PTRACE_SYSCALL single-stepping.
"""

from __future__ import annotations

import pyseccomp as seccomp


def install_trace_filter(
    syscalls: "list[str] | tuple[str, ...]",
    time_control: bool = False,
    time_realtime: bool = False,
) -> None:
    """Install a seccomp filter in the CALLING (traced) child, after
    PTRACE_TRACEME and before execve. The tracer must already have
    PTRACE_O_TRACESECCOMP set by then, or a filtered syscall fails with
    ENOSYS instead of trapping -- see agProxyPtrace.launch()'s SIGSTOP sync.

    With *time_control*, clock reads trap, and so do the timed waits that take an
    absolute deadline; relative waits never trap (see time_ctrl.py)."""
    filt = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in syscalls:
        filt.add_rule(seccomp.TRACE(0), name)
    if time_control:
        for name in ("clock_gettime", *(("gettimeofday", "time") if time_realtime else ())):
            filt.add_rule(seccomp.TRACE(0), name)
        absolute_flag = seccomp.Arg(1, seccomp.MASKED_EQ, 1, 1)
        filt.add_rule(seccomp.TRACE(0), "clock_nanosleep", absolute_flag)
        filt.add_rule(seccomp.TRACE(0), "timerfd_settime", absolute_flag)
        wait_bitset = seccomp.Arg(1, seccomp.MASKED_EQ, 0x7F, 9)
        filt.add_rule(seccomp.TRACE(0), "futex", wait_bitset, seccomp.Arg(3, seccomp.NE, 0))
    filt.load()
