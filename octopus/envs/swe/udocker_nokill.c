/* Preloaded into every dynamically linked process of a udocker trial (the trial's /etc/ld.so.preload, written by
 * UdockerEnvironment). P1 has no pid namespace and runs as us, so an agent's `pkill -9 python` once killed the host's
 * SGLang servers. Here kill, killpg, tgkill, tkill, sigqueue, pidfd_send_signal and syscall() for those (and
 * rt_sigqueueinfo, rt_tgsigqueueinfo) signal only processes
 * of the caller's own command: traced by the same PRoot (each command has its own, so concurrent commands of one trial,
 * e.g. fork branches' tool calls, are as foreign as the host), or zombies whose parent is; not PRoot's own child, the
 * wrapper that reports the command's status. Every other process acts as another user's would: EPERM for a pid,
 * skipped by a group kill (0, -pgrp), which signals the caller last and fails with EPERM only when it reached no one.
 * So `kill -9 0` or `pkill -9 bash` from a dynamically linked glibc process end the command but neither PRoot (whose
 * death would orphan setsid'd processes) nor the wrapper. Pids that do not exist get ESRCH, -1 EPERM; signal 0 passes
 * through.
 * Not covered: static or musl binaries, raw syscall instructions; e.g. a static busybox `kill -9 0` still kills PRoot,
 * and the command's setsid'd processes outlive it. Host processes stay visible (ps, /proc/<pid>/environ,
 * cmdline): keep secrets out of the environment and command lines of processes on trial hosts.
 * Agent tool calls also get no network: connect() (and syscall() for it) to an AF_INET/AF_INET6 address other than
 * loopback or 0.0.0.0/:: (both local), or to a local one on a common proxy port, fails with EACCES, connect(2)'s errno
 * for a local firewall rule, so a tool reads
 * "Permission denied" and does not retry as it would on ENETUNREACH. An agent once fetched the tested library's upstream
 * source (pip download, git clone, curl raw.githubusercontent.com), i.e. the reference fix. A tool call is a command
 * whose PRoot has OCTOPUS_NO_EGRESS=1 in its environment (UdockerEnvironment.exec env, set by HarborWorkspace.exec); the
 * command cannot change that, `unset` included. Healthcheck, verifier and setup keep the network (some tests download).
 * Not covered: glibc's resolver (its internal connect/sendto: DNS answers, no source), sendto/sendmsg (UDP, TCP fast
 * open), io_uring, ctypes through a libc handle (libc's own connect/syscall), a proxy on a port not listed, plus
 * everything not covered above.
 * x86_64 only; raw syscalls and no libc but errno, so one build loads under any glibc. */
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/syscall.h>

static long sys(long n, long a, long b, long c, long d, long e, long f) {
    register long r10 __asm__("r10") = d, r8 __asm__("r8") = e, r9 __asm__("r9") = f;
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    return r;
}

static void join(char *path, const char *pre, long num, const char *leaf) {
    char digits[24];
    int k = 0;
    while (*pre) *path++ = *pre++;
    do digits[k++] = (char)('0' + num % 10); while (num /= 10);
    while (k) *path++ = digits[--k];
    while (*leaf) *path++ = *leaf++;
    *path = 0;
}

/* the file <pre><num><leaf> into buf, NUL-terminated; its length, or -errno */
static long slurp(const char *pre, long num, const char *leaf, char *buf, long size) {
    char path[64];
    join(path, pre, num, leaf);
    long fd = sys(SYS_open, (long)path, 0, 0, 0, 0, 0), got = 0, r;
    if (fd < 0) return fd;
    while (got < size - 1 && (r = sys(SYS_read, fd, (long)(buf + got), size - 1 - got, 0, 0, 0)) > 0) got += r;
    sys(SYS_close, fd, 0, 0, 0, 0, 0);
    buf[got] = 0;
    return got;
}

/* what follows "<key>" and blanks in buf, or 0 */
static const char *at(const char *buf, const char *key) {
    for (; *buf; buf++) {
        int i = 0;
        while (key[i] && buf[i] == key[i]) i++;
        if (key[i]) continue;
        for (buf += i; *buf == ' ' || *buf == '\t'; buf++) {}
        return buf;
    }
    return 0;
}

static long num(const char *p) {
    long v = 0;
    for (; p && *p >= '0' && *p <= '9'; p++) v = v * 10 + (*p - '0');
    return v;
}

/* the PRoot tracing pid (a zombie: its parent), 0 if none, or -errno if pid's status cannot be read; *parent (if
 * given) = its parent */
static long tracer(long pid, long *parent) {
    char buf[4096];
    long n = slurp("/proc/", pid, "/status", buf, sizeof buf);
    if (n < 0) return n;
    const char *state = at(buf, "\nState:");
    long pp = num(at(buf, "\nPPid:"));
    if (parent) *parent = pp;
    if (state && *state == 'Z' && slurp("/proc/", pp, "/status", buf, sizeof buf) < 0)
        return 0;  /* untraced once dead */
    return num(at(buf, "\nTracerPid:"));
}

/* 0 when pid is traced by the PRoot tracing the caller and is not that PRoot's own child (the command's wrapper: it
 * reports the command's exit status, PRoot's own is garbage once it is killed), -ESRCH when pid does not exist, else
 * -EPERM */
static long ours(long pid) {
    long parent, its = tracer(pid, &parent), mine = tracer(sys(SYS_getpid, 0, 0, 0, 0, 0, 0), 0);
    return its == -ENOENT ? -ESRCH : mine > 0 && its == mine && parent != mine ? 0 : -EPERM;
}

/* kill(-pgrp, sig) for the members ours() allows, the caller last */
static long group(long pgrp, int sig) {
    char d[2048], buf[4096];
    long self = sys(SYS_getpid, 0, 0, 0, 0, 0, 0), in = 0, r = -ESRCH, n;
    long fd = sys(SYS_open, (long)"/proc", O_RDONLY | O_DIRECTORY, 0, 0, 0, 0);
    if (fd < 0) return -EPERM;
    while ((n = sys(SYS_getdents64, fd, (long)d, sizeof d, 0, 0, 0)) > 0)
        for (long i = 0; i < n; i += *(unsigned short *)(d + i + 16)) {  /* linux_dirent64: d_reclen, d_name */
            long pid = num(d + i + 19), o;
            if (!pid || slurp("/proc/", pid, "/status", buf, sizeof buf) < 0 || num(at(buf, "\nNSpgid:")) != pgrp)
                continue;
            if (pid == self) in = 1;
            else if ((o = ours(pid)) == -EPERM) r = r ? -EPERM : 0;
            else if (!o && !sys(SYS_kill, pid, sig, 0, 0, 0, 0)) r = 0;
        }
    sys(SYS_close, fd, 0, 0, 0, 0, 0);
    return in && !sys(SYS_kill, self, sig, 0, 0, 0, 0) ? 0 : r;
}

/* kill(pid, sig) as if every process ours() rejects were another user's */
static long deliver(int pid, int sig) {
    if (sig == 0) return sys(SYS_kill, pid, 0, 0, 0, 0, 0);
    if (pid == -1) return -EPERM;
    if (pid <= 0) return group(pid ? -(long)pid : sys(SYS_getpgid, 0, 0, 0, 0, 0, 0), sig);
    long r = ours(pid);
    return r ? r : sys(SYS_kill, pid, sig, 0, 0, 0, 0);
}

/* 0 when the thread or pidfd target pid may get sig (tgkill and tkill reject pid <= 0 themselves), else -errno */
static long check(int pid, int sig) {
    return sig == 0 || pid <= 0 ? 0 : ours(pid);
}

/* a pidfd (its fdinfo has Pid:) or an fd of a /proc/<pid> directory */
static long check_pidfd(int fd, int sig) {
    char path[64], buf[1024];
    if (slurp("/proc/self/fdinfo/", fd, "", buf, sizeof buf) < 0) return -EPERM;
    const char *pid = at(buf, "\nPid:");
    if (!pid) {
        join(path, "/proc/self/fd/", fd, "");
        long n = sys(SYS_readlink, (long)path, (long)buf, sizeof buf - 1, 0, 0, 0);
        buf[n > 0 ? n : 0] = 0;
        pid = at(buf, "/proc/");
        if (pid != buf + 6 || *pid < '1' || *pid > '9') return -EPERM;
    }
    return check((int)num(pid), sig);
}

/* an AF_INET/AF_INET6 address off this host (not 127/8, 0.0.0.0, ::1, ::, or those IPv4-mapped), or a local one on a
 * common proxy port: trial hosts run xray (10808/10809) and privoxy (8118) on loopback, and udocker has no network
 * namespace, so the host's loopback is the container's */
static int zeros(const unsigned char *b, int n) {
    while (n--)
        if (b[n]) return 0;
    return 1;
}

static int remote(const struct sockaddr *a, long len) {
    const unsigned char *ip;
    int local;
    if (a && a->sa_family == AF_INET && len >= (long)sizeof(struct sockaddr_in)) {
        ip = (const unsigned char *)&((const struct sockaddr_in *)a)->sin_addr;
        local = ip[0] == 127 || zeros(ip, 4);
    } else if (a && a->sa_family == AF_INET6 && len >= 24) {  /* SIN6_LEN_RFC2133: the kernel takes it, no scope id */
        ip = ((const struct sockaddr_in6 *)a)->sin6_addr.s6_addr;
        local = zeros(ip, 10) && ip[10] == 0xff && ip[11] == 0xff ? ip[12] == 127 || zeros(ip + 12, 4)
                : zeros(ip, 15) && ip[15] <= 1;
    } else
        return 0;
    const unsigned char *port = (const unsigned char *)a + 2;  /* sin_port, sin6_port: big-endian */
    int p = port[0] << 8 | port[1];
    return !local || p == 10808 || p == 10809 || p == 8118 || p == 7890 || p == 7891 || p == 1080;
}

/* -EACCES when the caller's PRoot runs an agent command (OCTOPUS_NO_EGRESS=1 in its environment) and a is remote */
static long egress(const struct sockaddr *a, long len) {
    static const char key[] = "OCTOPUS_NO_EGRESS=1";
    char buf[16384];
    long pid = remote(a, len) ? tracer(sys(SYS_getpid, 0, 0, 0, 0, 0, 0), 0) : 0;
    long n = pid > 0 ? slurp("/proc/", pid, "/environ", buf, sizeof buf) : -1;
    for (long i = 0; i < n; i++) {  /* NUL-separated entries */
        int k = 0;
        while (key[k] && buf[i + k] == key[k]) k++;
        if (!key[k] && !buf[i + k]) return -EACCES;
        while (buf[i]) i++;
    }
    return 0;
}

static long done(long r) {
    if ((unsigned long)r > -4096UL) {
        errno = (int)-r;
        return -1;
    }
    return r;
}

int kill(int pid, int sig) {
    return (int)done(deliver(pid, sig));
}

int killpg(int pgrp, int sig) {
    return pgrp < 0 ? (int)done(-EINVAL) : kill(-pgrp, sig);
}

int tgkill(int tgid, int tid, int sig) {
    long r = check(tgid, sig);
    return (int)done(r ? r : sys(SYS_tgkill, tgid, tid, sig, 0, 0, 0));
}

int tkill(int tid, int sig) {
    long r = check(tid, sig);
    return (int)done(r ? r : sys(SYS_tkill, tid, sig, 0, 0, 0, 0));
}

/* procps's `pkill --queue` */
int sigqueue(int pid, int sig, const union sigval value) {
    long r = check(pid, sig);
    if (r) return (int)done(r);
    siginfo_t info;  /* what glibc's sigqueue sends; zeroed by hand, -nostdlib has no memset */
    for (unsigned long i = 0; i < sizeof info; i++) ((volatile char *)&info)[i] = 0;
    info.si_signo = sig;
    info.si_code = SI_QUEUE;
    info.si_pid = (int)sys(SYS_getpid, 0, 0, 0, 0, 0, 0);
    info.si_uid = (unsigned)sys(SYS_getuid, 0, 0, 0, 0, 0, 0);
    info.si_value = value;
    return (int)done(sys(SYS_rt_sigqueueinfo, pid, sig, (long)&info, 0, 0, 0));
}

/* glibc >= 2.36 */
int pidfd_send_signal(int pidfd, int sig, siginfo_t *info, unsigned int flags) {
    long r = check_pidfd(pidfd, sig);
    return (int)done(r ? r : sys(SYS_pidfd_send_signal, pidfd, sig, (long)info, flags, 0, 0));
}

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    long r = egress(addr, len);
    return (int)done(r ? r : sys(SYS_connect, fd, (long)addr, len, 0, 0, 0));
}

/* psmisc's killall and CPython's signal.pidfd_send_signal call syscall() */
long syscall(long n, ...) {
    va_list ap;
    va_start(ap, n);
    long a = va_arg(ap, long), b = va_arg(ap, long), c = va_arg(ap, long);
    long d = va_arg(ap, long), e = va_arg(ap, long), f = va_arg(ap, long);
    va_end(ap);
    if (n == SYS_kill) return done(deliver((int)a, (int)b));
    long r = n == SYS_tkill || n == SYS_rt_sigqueueinfo ? check((int)a, (int)b)
             : n == SYS_tgkill || n == SYS_rt_tgsigqueueinfo ? check((int)a, (int)c)
             : n == SYS_pidfd_send_signal ? check_pidfd((int)a, (int)b)
             : n == SYS_connect ? egress((const struct sockaddr *)b, c) : 0;
    return done(r ? r : sys(n, a, b, c, d, e, f));
}
