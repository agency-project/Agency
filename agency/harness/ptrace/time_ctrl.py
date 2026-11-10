"""Virtual monotonic clock for a ptrace-supervised process tree.

While every task is parked the clock stands still: reads return real time minus the total
frozen time, and timers armed in the kernel are disarmed for the freeze and re-armed after.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
import time

from . import _ctypes_defs as pt

_SYSCALL = pt.SYSCALL_NUMBERS
_MONOTONIC_CLOCKS = frozenset({1, 4, 6, 7})
_REALTIME_CLOCKS = frozenset({0, 5})
_NS_PER_SECOND = 10**9
_NO_SYSCALL = 2**64 - 1  # orig_rax value that makes the kernel skip the syscall
# Absolute-deadline waits: syscall -> (register holding the user pointer, struct size, timespec offset).
_ABSOLUTE_DEADLINE = {
    _SYSCALL["clock_nanosleep"]: ("rdx", 16, 0),
    _SYSCALL["futex"]: ("r10", 16, 0),
    _SYSCALL["timerfd_settime"]: ("rdx", 32, 16),
}
# Timed waits the kernel restarts on its own after a stop, against a stale real-time deadline.
_RESTARTED_TIMED_WAITS = frozenset(
    _SYSCALL[name] for name in ("poll", "nanosleep", "futex", "restart_syscall", "clock_nanosleep")
)


class TimeController:
    """Owns the virtual clock offset and the timer bookkeeping for one traced tree."""

    def __init__(self, virtualize_realtime: bool = False) -> None:
        self.paused_ns = 0
        self.frozen = False
        self.virtualize_realtime = virtualize_realtime
        self._virtual_clocks = _MONOTONIC_CLOCKS | (
            _REALTIME_CLOCKS if virtualize_realtime else frozenset()
        )
        self._frozen_at_ns = 0
        self._disarmed_timers: "list[tuple[int, int, int, int, int]]" = []

    def on_exec(self, pid: int) -> None:
        """Hide the vDSO from the new image so libc and Go fall back to real, trappable syscalls."""
        address = pt.get_regs(pid).rsp
        argument_count = struct.unpack("<Q", pt.read_bytes(pid, address, 8))[0]
        address += 8 * (argument_count + 2)  # skip argc, argv[] and its NULL
        while struct.unpack("<Q", pt.read_bytes(pid, address, 8))[0]:
            address += 8
        address += 8
        while True:
            key = struct.unpack("<QQ", pt.read_bytes(pid, address, 16))[0]
            if key == 0:
                return
            if key == 33:  # AT_SYSINFO_EHDR becomes AT_IGNORE
                pt.write_bytes(pid, address, struct.pack("<QQ", 1, 0))
                return
            address += 16

    def on_trap(self, pid: int, regs: "pt.UserRegsStruct") -> None:
        """Answer a trapped clock read, or move an absolute deadline from virtual to real time."""
        syscall = regs.orig_rax
        return_value = None
        if syscall == _SYSCALL["clock_gettime"] and regs.rdi in self._virtual_clocks:
            virtual_ns = time.clock_gettime_ns(regs.rdi) - self.paused_ns
            pt.write_bytes(pid, regs.rsi, struct.pack("<qq", *divmod(virtual_ns, _NS_PER_SECOND)))
            return_value = 0
        elif syscall == _SYSCALL["gettimeofday"]:
            virtual_ns = time.time_ns() - self.paused_ns
            if regs.rdi:
                seconds, nanoseconds = divmod(virtual_ns, _NS_PER_SECOND)
                pt.write_bytes(pid, regs.rdi, struct.pack("<qq", seconds, nanoseconds // 1000))
            return_value = 0
        elif syscall == _SYSCALL["time"]:
            return_value = (time.time_ns() - self.paused_ns) // _NS_PER_SECOND
            if regs.rdi:
                pt.write_bytes(pid, regs.rdi, struct.pack("<q", return_value))
        elif syscall in _ABSOLUTE_DEADLINE and self.paused_ns:
            if syscall == _SYSCALL["clock_nanosleep"]:
                clock = regs.rdi
            elif syscall == _SYSCALL["futex"]:
                clock = 0 if regs.rsi & 256 else 1  # FUTEX_CLOCK_REALTIME
            else:
                try:
                    with open(f"/proc/{pid}/fdinfo/{regs.rdi}", encoding="utf-8") as fdinfo:
                        clock = int(re.search(r"clockid:\s*(\d+)", fdinfo.read())[1])
                except (OSError, TypeError):
                    return  # bad descriptor: let the kernel fail the call itself
            if clock in self._virtual_clocks:
                pointer_register, size, deadline_offset = _ABSOLUTE_DEADLINE[syscall]
                user_struct = bytearray(pt.read_bytes(pid, getattr(regs, pointer_register), size))
                seconds, nanoseconds = struct.unpack_from("<qq", user_struct, deadline_offset)
                real_ns = seconds * _NS_PER_SECOND + nanoseconds + self.paused_ns
                struct.pack_into(
                    "<qq", user_struct, deadline_offset, *divmod(real_ns, _NS_PER_SECOND)
                )
                scratch_address = (regs.rsp - 1024) & ~15
                pt.write_bytes(pid, scratch_address, bytes(user_struct))
                setattr(regs, pointer_register, scratch_address)
                pt.set_regs(pid, regs)
        if return_value is not None:
            regs.orig_rax, regs.rax = _NO_SYSCALL, return_value
            pt.set_regs(pid, regs)

    def freeze(self, injectable_pids: "list[int]") -> None:
        """Stop the clock. Call with every task parked; *injectable_pids* are those parked outside a syscall."""
        injection_pid_by_tgid: "dict[int, int]" = {}
        for pid in injectable_pids:
            regs = pt.get_regs(pid)
            restart_code = -ctypes.c_longlong(regs.rax).value
            if regs.orig_rax in _RESTARTED_TIMED_WAITS and restart_code in (516, 514):
                # The kernel would restart this wait against a stale deadline; EINTR makes the program retry.
                regs.rax, regs.orig_rax = (-4) & _NO_SYSCALL, _NO_SYSCALL
                pt.set_regs(pid, regs)
            with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
                tgid = next(
                    int(line.split()[1]) for line in status_file if line.startswith("Tgid:")
                )
            injection_pid_by_tgid.setdefault(tgid, pid)

        self._disarmed_timers = []
        for tgid, pid in injection_pid_by_tgid.items():
            timers = [(_SYSCALL["setitimer"], 0)]
            try:
                timers += [
                    (_SYSCALL["timerfd_settime"], int(fd))
                    for fd in os.listdir(f"/proc/{tgid}/fd")
                    if os.readlink(f"/proc/{tgid}/fd/{fd}") == "anon_inode:[timerfd]"
                ]
                with open(f"/proc/{tgid}/timers", encoding="utf-8") as timer_list:
                    timers += [
                        (_SYSCALL["timer_settime"], int(line.split()[1]))
                        for line in timer_list
                        if line.startswith("ID: ")
                    ]
            except OSError:
                pass
            for syscall, selector in timers:
                unit_ns = 1000 if syscall == _SYSCALL["setitimer"] else 1
                expired_unread = armed_before = False
                try:
                    if syscall == _SYSCALL["timerfd_settime"]:
                        # timerfd_settime() discards an expiration nobody has read yet.
                        with open(f"/proc/{tgid}/fdinfo/{selector}", encoding="utf-8") as fdinfo:
                            text = fdinfo.read()
                        expired_unread = int(re.search(r"ticks:\s*(\d+)", text)[1]) > 0
                        current_value, current_interval = [
                            tuple(map(int, pair)) for pair in re.findall(r"\((\d+), (\d+)\)", text)
                        ]
                        armed_before = any(current_value)
                        if expired_unread and not any(current_interval):
                            continue
                    old = struct.unpack("<qqqq", self._inject(pid, syscall, selector, bytes(32)))
                except (OSError, pt.PtraceError):
                    continue
                interval_ns = old[0] * _NS_PER_SECOND + old[1] * unit_ns
                value_ns = old[2] * _NS_PER_SECOND + old[3] * unit_ns
                if expired_unread or (armed_before and value_ns == 0):
                    value_ns = 1  # expired meanwhile: keep one expiration pending
                elif value_ns == 0:
                    value_ns = interval_ns  # periodic timer caught while reloading
                if value_ns > 0:
                    self._disarmed_timers.append((pid, syscall, selector, interval_ns, value_ns))
        self._frozen_at_ns = time.monotonic_ns()
        self.frozen = True

    def thaw(self) -> None:
        """Restart the clock and re-arm the timers. Call before any task resumes."""
        self.paused_ns += time.monotonic_ns() - self._frozen_at_ns
        for pid, syscall, selector, interval_ns, value_ns in self._disarmed_timers:
            unit_ns = 1000 if syscall == _SYSCALL["setitimer"] else 1
            fields = []
            for total_ns in (interval_ns, value_ns):
                fields += [total_ns // _NS_PER_SECOND, total_ns % _NS_PER_SECOND // unit_ns]
            if fields[2] == fields[3] == 0:
                fields[3] = 1
            try:
                self._inject(pid, syscall, selector, struct.pack("<qqqq", *fields))
            except (OSError, pt.PtraceError):
                pass
        self._disarmed_timers = []
        self.frozen = False

    def _inject(self, pid: int, syscall: int, selector: int, new_value: bytes) -> bytes:
        """Run one timer syscall inside parked task *pid*; return the timer's previous 32-byte value."""
        saved_regs = pt.get_regs(pid)
        scratch_address = (saved_regs.rsp - 4096) & ~15
        pt.write_bytes(pid, scratch_address, new_value)
        # Block every signal meanwhile: a pending one would push a handler frame onto the stack.
        previous_mask, all_signals = ctypes.c_uint64(), ctypes.c_uint64(2**64 - 1)
        pt.ptrace(pt.PTRACE_GETSIGMASK, pid, 8, ctypes.addressof(previous_mask))
        pt.ptrace(pt.PTRACE_SETSIGMASK, pid, 8, ctypes.addressof(all_signals))
        injected_regs = pt.get_regs(pid)
        injected_regs.rax, injected_regs.orig_rax = syscall, _NO_SYSCALL
        old_address = scratch_address + 32
        injected_regs.rdi, injected_regs.rsi, injected_regs.rdx, injected_regs.r10 = (
            (selector, scratch_address, old_address, 0)
            if syscall == _SYSCALL["setitimer"]
            else (selector, 0, scratch_address, old_address)
        )
        with open(f"/proc/{pid}/mem", "r+b", buffering=0) as task_memory:
            task_memory.seek(saved_regs.rip)
            original_code = task_memory.read(2)
            try:
                task_memory.seek(saved_regs.rip)
                task_memory.write(b"\x0f\x05")  # a `syscall` instruction at the resume point
                pt.set_regs(pid, injected_regs)
                for _ in range(6):
                    pt.ptrace(pt.PTRACE_SINGLESTEP, pid, 0, 0)
                    _, status = os.waitpid(pid, pt.WAIT_ALL)
                    if (status >> 16) not in (pt.PTRACE_EVENT_SECCOMP, pt.PTRACE_EVENT_STOP):
                        break
                previous_value = pt.read_bytes(pid, old_address, 32)
            finally:
                task_memory.seek(saved_regs.rip)
                task_memory.write(original_code)
                # Leave the task as the kernel would after a handler-less stop, emulating a pending restart.
                pending_result = ctypes.c_longlong(saved_regs.rax).value
                if ctypes.c_longlong(saved_regs.orig_rax).value >= 0:
                    if pending_result in (-512, -513, -514):
                        saved_regs.rax, saved_regs.rip = saved_regs.orig_rax, saved_regs.rip - 2
                    elif pending_result == -516:
                        saved_regs.rax = _SYSCALL["restart_syscall"]
                        saved_regs.rip -= 2
                saved_regs.orig_rax = _NO_SYSCALL
                pt.set_regs(pid, saved_regs)
                pt.ptrace(pt.PTRACE_SETSIGMASK, pid, 8, ctypes.addressof(previous_mask))
        return previous_value
