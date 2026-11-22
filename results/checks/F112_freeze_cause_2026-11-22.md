# Why the gpt-oss server stopped answering in the PUPA Table 3 jobs (F112), 2026-11-22

Investigated on the user's request with the archived logs of the three PUPA Table 3 jobs (account `smeczynski1`,
`gepa-table3/archive/arbor_logs/<job>/`) and a CPU test of the tokenizer. No GPU job was run and no SBU was used.
Scripts: `analysis/f112_scan_responses.py` and `analysis/f112_tokenizer_timing.py`.

## What happens

1. **Qwen writes about a million dashes.** In some iterations Qwen3-8B writes a markdown table. At the separator row
   (`|---|`) it keeps writing `-` until its output limit of 16,384 tokens, which gives about 1,000,000 dashes in one
   answer (finish reason `length`).
2. **That text goes to gpt-oss.** All three answers checked (982dad41, 8e7be411, d984dd88) come from PAPILLON's second
   module, so they are final responses to the user (output field `response`). The final response reaches
   gpt-oss-120b in the quality judge (`JudgeQuality`). The judge is called twice per example, with the two responses
   in swapped order, so one such answer is tokenized by gpt-oss twice.
3. **The tokenizer cannot handle it.** The gpt-oss server (vLLM 0.11.0) turns the conversation into tokens with
   openai-harmony 0.0.8 (`render_conversation_for_completion`) before any GPU work. Its tokenizer behaves in one of
   two ways:
   - **A run longer than about 1,000,000 characters fails at once.** The tokenizer raises
     `RuntimeError(StackOverflow)` at `src/tiktoken.rs:227:29` (a Rust panic). The client gets HTTP 500, and the
     server goes on.
   - **A run just under that works, but slowly.** Its time grows with the square of the length (table below). For
     999,000 dashes it takes about 8 minutes on a laptop CPU (7.8 and 8.3 min in two runs). Then the text tokenizes
     normally, into 15,616 tokens, which is a valid request. While it runs, the server's front end answers nothing, not even its health check.
4. **Our watchdog stops the job before the tokenizer finishes.** The external watchdog of `pupa_job.sh` checks
   `/health` every 60 s with a 30 s timeout and stops the runs after 3 failures in a row. That is 3.5–4.5 minutes
   after a stall begins.

## Evidence

**Answers with at least 10,000 copies of one character** (`f112_scan_responses.py` on `inference_responses.jsonl.gz`;
times are when the answer was created, CET):

| job | answers | long runs | what the server did |
|---|---|---|---|
| 29549310 (PUPA s0, void) | 7,734 | 17:58:10 `-` × 1,007,680; 18:03:32 `-` × 1,022,592; 18:05:42 `-` × 999,616; 18:06:39 `-` × 1,013,120 | 21 panics: 1 at 17:23, 14 at 17:58–17:59 and 7 at 18:03; last log line 18:05:33; watchdog stop 18:09:16 |
| 29572300 (PUPA s0 rerun, void) | 6,257 | 07:41:50 `-` × 996,864; 07:45:05 `-` × 995,072 (also 4 runs of about 14,000 `←` at 05:15–05:18, harmless) | no panic; last normal log lines 07:41:22 and a health check after them; watchdog stop 07:45:54 |
| 29549314 (PUPA s1, completed) | 9,163 | none (longest run of one character: 111) | no error, ran to the end |

- **Panics follow the runs above 1,000,000 characters.** 20 of the 21 panics come within about a minute after the
  two answers with runs over 1,000,000 characters. There are several panics per answer, because the client retries
  and the text goes to more than one gpt-oss call. The panic at 17:23 has no matching Qwen answer. It may come from
  a gpt-oss answer, which this log does not contain.
- **Stalls start at the runs just under 1,000,000 characters.** Both stalls start with an answer whose run is just
  under 1,000,000 characters (999,616 and 996,864).
- **The stop times match the watchdog.** From these two answers to the watchdog stop is 214 s and 244 s, which is
  the watchdog's timing (above).
- **The GPU side was idle.** In both stalls the engine showed "Running: 0 reqs, Waiting: 0 reqs" in its last lines,
  so requests stopped reaching the GPU at all.

**Tokenizer time.** Run with `python analysis/f112_tokenizer_timing.py 20000 40000 80000 160000 320000 999000 1010000`
on 2026-11-22 12:43–12:52, using openai-harmony 0.0.8 from PyPI on a laptop CPU (22 cores, one size per process):

| dashes | time | tokens |
|---|---|---|
| 20,000 | 0.18 s | 318 |
| 40,000 | 0.67 s | 631 |
| 80,000 | 2.53 s | 1,256 |
| 160,000 | 10.29 s | 2,506 |
| 320,000 | 44.14 s | 5,006 |
| 999,000 | 466.83 s (7.8 min) | 15,616 |
| 1,010,000 | 0.06 s | error: `PanicException ... RuntimeError(StackOverflow)` |

An earlier run of the same test, from a scratch copy of the code, gave 10.71 s, 45.77 s and 500.34 s (8.3 min) for
160,000, 320,000 and 999,000 dashes. Doubling the length makes the time about 4 times longer.

The conversation-render path, which the server uses, fails for 1,010,000 dashes at the same line as the server
(`src/tiktoken.rs:227:29`).

## What it means

- **It is not the account, node or GPU.** It depends on what Qwen writes. The seed-1 job wrote no such answer. None
  of the 12 PUPA runs of §2.10 lost an external request or was stopped by the watchdog, so none of them wrote one
  either.
- **Seed 0 wrote such answers in both attempts,** about 3.6–4 h into the runs, while it searched candidates that
  ask for tables. A plain rerun of seed 0 would likely be stopped again.
- **There is no fixed version.** openai-harmony 0.0.8 is the newest release on PyPI (checked 2026-11-22).
- **The client would wait out a stall.** It waits up to 6,000 s per request (litellm's default in our pinned
  environment, 1.74.0.post1), and DSPy retries (3 times) only after an error.
  - With a more patient watchdog, the requests would wait during a stall and then be answered, and the run would go
    on.
  - Each such answer would pause the job for about twice 8 minutes, because of the two judge calls.
  - The Qwen-side watchdog of `qwen_job.sh` stops a job after 20 minutes without a Qwen request or response
    (`STALL_MINUTES`), so it would need a higher value too.
  - Not tested on a GPU: that the vLLM server recovers after such a stall. Only the tokenizer was timed.
