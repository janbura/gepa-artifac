# F112: time the gpt-oss tokenizer (openai-harmony, as used by the vLLM 0.11.0 gpt-oss server) on the text Qwen
# wrote in the frozen PUPA jobs: a markdown table row followed by n dashes. Each size runs in its own process with
# a time limit, so a size that does not finish cannot block the others.
#   python analysis/f112_tokenizer_timing.py [sizes ...]          (needs openai-harmony==0.0.8, CPU only)
# openai-harmony downloads its vocabulary file once; set TIKTOKEN_RS_CACHE_DIR to choose where it is kept.
import sys, subprocess

CHILD = r'''
import sys, time
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
n = int(sys.argv[1])
text = "| # | Prompt |\n|" + "-" * n
t0 = time.time()
try:
    toks = enc.encode(text, allowed_special=set())
    print(f"{time.time() - t0:.2f} s, {len(toks):,} tokens")
except BaseException as e:  # a Rust panic arrives as PanicException, which is not an Exception
    print(f"{time.time() - t0:.2f} s, error {type(e).__name__}: {str(e)[:90]}")
'''

sizes = [int(x) for x in sys.argv[1:]] or [20000, 40000, 80000, 160000, 320000, 1010000]
LIMIT = 1500
for n in sizes:
    try:
        out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(n)], capture_output=True, text=True,
                             timeout=LIMIT)
        lines = [x for x in out.stdout.splitlines() if x.strip()]
        print(f"n = {n:>9,}: {lines[-1] if lines else out.stderr.strip()[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"n = {n:>9,}: not finished after {LIMIT} s", flush=True)
