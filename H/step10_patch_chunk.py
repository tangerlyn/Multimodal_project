"""Make --batch_items a real upper bound: split a subject's items into chunks of that size, so a long subject
(MMMU test has up to ~550 items per subject) is written to disk chunk by chunk instead of all at the end.
Per-item seeds (request_seed) make each item's output independent of how items are batched. Idempotent.
  python step10_patch_chunk.py /content/MMD/src/eval_mmmu.py
"""
import sys
p = sys.argv[1]
s = open(p).read()
old = """            if pending and (len(pending) >= args.batch_items or last):
                flush(pending, fout)
                pending = []
"""
new = """            while len(pending) >= args.batch_items:   # chunked (patch_chunk.py)
                flush(pending[:args.batch_items], fout)
                pending = pending[args.batch_items:]
            if pending and last:
                flush(pending, fout)
                pending = []
"""
if new not in s:
    assert s.count(old) == 1, "flush block not found"
    s = s.replace(old, new)
    open(p, "w").write(s)
print("chunk patch present:", new in open(p).read())
