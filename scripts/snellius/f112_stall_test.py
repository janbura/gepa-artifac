"""Test of the F112 workaround on a GPU, run as the one command of a PUPA job (plan 2.14 addendum of 2026-11-24).

Sends the gpt-oss server the kind of text that stalled the two void PUPA seed-0 jobs (a markdown table
separator with about a million dashes, see results/checks/F112_freeze_cause_2026-11-22.md) and checks that the
server answers it, answers normal requests afterwards, and survives a run longer than a million characters
(which its tokenizer rejects with an error). While each long request is in flight, a helper thread polls
/health and sends one normal request, like the other run threads would.

The server address comes from the job (PUPA_EXTERNAL_API_BASE, PUPA_EXTERNAL_MODEL). Standard library only.
Usage: python -I scripts/snellius/f112_stall_test.py --out <report.md>
Exit code 0 = every criterion passed, 1 = not.
"""
import argparse
import http.client
import json
import os
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

PREFIX = "| # | Prompt |\n|"
LONG_RUN = 999_000  # just under the size at which the tokenizer fails at once (laptop: about 8 min)
OVER_RUN = 1_010_000  # over it: the tokenizer raises RuntimeError(StackOverflow), the server returns an error


def now():
    return datetime.now().strftime("%H:%M:%S")


class Server:
    def __init__(self, api_base, model):
        u = urlparse(api_base)
        self.host, self.port = u.hostname, u.port
        self.path = u.path.rstrip("/")  # /v1
        self.model = model

    def chat(self, content, max_tokens=16, timeout=2400):
        body = json.dumps({"model": self.model, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": content}]})
        t0 = time.time()
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
            c.request("POST", self.path + "/chat/completions", body=body,
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read().decode("utf-8", "replace")
            status = r.status
            c.close()
        except Exception as e:  # timeout, connection refused, ...
            status, data = -1, f"{type(e).__name__}: {e}"
        return status, time.time() - t0, data[:200].replace("\n", " ")

    def health(self, timeout=30):
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
            c.request("GET", "/health")
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except Exception:
            return False


class Watcher(threading.Thread):
    """Polls /health every 30 s and sends one normal request 60 s after the start, until stopped."""

    def __init__(self, server):
        super().__init__(daemon=True)
        self.s = server
        self.stop = threading.Event()
        self.health = []  # (seconds since start, ok)
        self.side = None  # (status, seconds, text) of the normal request sent during the long one

    def run(self):
        t0 = time.time()
        sent = False
        while not self.stop.is_set():
            self.health.append((round(time.time() - t0), self.s.health()))
            if not sent and time.time() - t0 >= 60:
                sent = True
                threading.Thread(target=self._side, daemon=True).start()
            self.stop.wait(30)

    def _side(self):
        self.side = self.s.chat("Say ok.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--long", type=int, default=LONG_RUN)
    ap.add_argument("--over", type=int, default=OVER_RUN)
    a = ap.parse_args()
    base = os.environ["PUPA_EXTERNAL_API_BASE"]
    model = os.environ["PUPA_EXTERNAL_MODEL"].split("/", 1)[-1]
    s = Server(base, model)
    rows = []  # (step, status, seconds, note)
    verdicts = []  # (criterion, passed)

    def write():
        with open(a.out, "w") as f:
            f.write(f"# F112 stall test on {base} ({model}), {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            f.write("| step | HTTP status | seconds | note |\n|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]} |\n")
            f.write("\n| criterion | passed |\n|---|---|\n")
            for c, ok in verdicts:
                f.write(f"| {c} | {'yes' if ok else 'NO'} |\n")
            f.write(f"\nOverall: {'PASS' if all(ok for _, ok in verdicts) else 'FAIL'}\n")

    def step(name, content, **kw):
        print(f"{now()} {name} ...", flush=True)
        st, sec, txt = s.chat(content, **kw)
        rows.append((name, st, sec, txt))
        print(f"{now()} {name}: status {st} after {sec:.1f} s: {txt[:100]}", flush=True)
        write()
        return st, sec

    def normal(name):
        st, _ = step(name, "Say ok.")
        verdicts.append((f"{name}: normal request answered (HTTP 200)", st == 200))
        write()

    def long_request(name, n):
        w = Watcher(s)
        w.start()
        st, sec = step(name, PREFIX + "-" * n)
        w.stop.set()
        w.join(5)
        for _ in range(60):  # give the side request up to a minute more to come back
            if w.side is not None:
                break
            time.sleep(1)
        fails = [t for t, ok in w.health if not ok]
        rows.append((f"{name}: health checks during it", len(w.health), 0.0,
                     f"{len(fails)} failed" + (f" (first at {fails[0]} s, last at {fails[-1]} s)" if fails else "")))
        side = w.side or (-1, 0.0, "no answer within 60 s after the long request")
        rows.append((f"{name}: normal request sent 60 s into it", side[0], side[1], side[2]))
        write()
        return st, sec, side[0]

    def recovered(name):
        t0 = time.time()
        ok = False
        while time.time() - t0 < 120:
            if s.health():
                ok = True
                break
            time.sleep(5)
        rows.append((f"{name}: health check after it", 200 if ok else -1, time.time() - t0, "ok" if ok else "no answer within 120 s"))
        verdicts.append((f"{name}: server answers its health check within 120 s afterwards", ok))
        write()

    normal("S1 normal request")
    st, sec, side = long_request("S2 long request", a.long)
    verdicts.append((f"S2: request with {a.long:,} dashes answered (HTTP 200)", st == 200))
    verdicts.append(("S2: normal request sent during it answered (HTTP 200)", side == 200))
    recovered("S2")
    normal("S3 normal request after the long one")
    st, sec = step("S4 over-long request", PREFIX + "-" * a.over)
    verdicts.append((f"S4: request with {a.over:,} dashes gets an answer or an error within 300 s (no hang)", st != -1 and sec < 300))
    recovered("S4")
    normal("S5 normal request after the over-long one")
    st, sec, side = long_request("S6 second long request", a.long)
    verdicts.append((f"S6: second request with {a.long:,} dashes answered (HTTP 200)", st == 200))
    verdicts.append(("S6: normal request sent during it answered (HTTP 200)", side == 200))
    recovered("S6")
    normal("S7 final normal request")
    write()
    passed = all(ok for _, ok in verdicts)
    print(f"{now()} overall: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok in verdicts)}/{len(verdicts)} criteria)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
