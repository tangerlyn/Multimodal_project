"""Add MMMU_IMAGE_MODE=none to src/eval_mmmu.py: the images are dropped, the text (including the literal
`<image N>` tokens, as in the 'first' image position) is unchanged. Recorded in run_meta.json. Idempotent.
  python step07_patch_noimg.py /content/MMD/src/eval_mmmu.py
"""
import sys
p = sys.argv[1]; s = open(p).read()
if "MMMU_IMAGE_MODE" not in s:
    a = 's["images"], args.max_pixels, args.min_pixels,'
    b = '"sampling": {**SAMPLING_RECIPE'
    assert s.count(a) == 1 and s.count(b) == 1, (s.count(a), s.count(b))
    s = s.replace(a, '({} if os.environ.get("MMMU_IMAGE_MODE") == "none" else s["images"]), args.max_pixels, args.min_pixels,')
    s = s.replace(b, '"image_mode": os.environ.get("MMMU_IMAGE_MODE", "normal"), ' + b)
    open(p, "w").write(s)
print("image-mode patch present:", "MMMU_IMAGE_MODE" in open(p).read())
