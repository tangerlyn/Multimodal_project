"""Build one full Qwen3-VL checkpoint that vLLM (MMMU eval) can load directly.

  IAA (LLM LoRA only):
    python step06_merge_model.py --lora ~/mmdl/runs/IAA_r16_lr1e-4_s3407/lora --out ~/mmdl/merged/IAA
  Both (connector + LLM LoRA):
    python step06_merge_model.py --lora RUN/lora --connector RUN/connector.pt --out ~/mmdl/merged/EBoth
  Connector only (B checkpoints are already full models; this is for a bare weights file):
    python step06_merge_model.py --connector RUN/trainable_weights.pt --out ~/mmdl/merged/B2

Then:  bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --model_path ~/mmdl/merged/IAA ...

--connector is a torch.save'd {parameter name: tensor} dict, names as in model.named_parameters()
(e.g. model.visual.merger.linear_fc1.weight), which is what train_merger.py writes.
After saving, the script reloads the checkpoint and checks:
  * the ViT is bit-identical to the base model (it must never change),
  * every LoRA-targeted LLM weight / every connector weight actually differs from the base,
  * everything else is identical.
"""

import argparse
import json
from pathlib import Path

from step00_common import BASE_MODEL, BASE_REVISION

CONNECTOR_KEYS = ("visual.merger.", "visual.deepstack_merger_list.")


def is_connector(name):
    return any(k in name for k in CONNECTOR_KEYS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_MODEL, help="HF id or a local full checkpoint")
    ap.add_argument("--revision", default=BASE_REVISION, help="ignored for a local --base")
    ap.add_argument("--lora", type=Path, help="PEFT adapter folder (train_iaa.py lora/ or lora_last/)")
    ap.add_argument("--connector", type=Path, help="connector weights .pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    if not args.lora and not args.connector:
        ap.error("give --lora and/or --connector")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output not empty: {args.out}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rev = None if Path(args.base).exists() else args.revision
    load = dict(revision=rev, dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base, **load)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}   # CPU copy, ~8 GB bf16

    lora_targets = set()
    if args.lora:
        from peft import PeftModel
        peft = PeftModel.from_pretrained(model, str(args.lora))
        lora_targets = {n.split(".lora_A")[0].replace("base_model.model.", "", 1) + ".weight"
                        for n, _ in peft.named_parameters() if ".lora_A" in n}
        model = peft.merge_and_unload()
        print(f"merged LoRA {args.lora}: {len(lora_targets)} linear layers")

    conn_names = set()
    if args.connector:
        sd = torch.load(args.connector, map_location="cpu")
        params = dict(model.named_parameters())
        unknown = [k for k in sd if k not in params]
        not_conn = [k for k in sd if not is_connector(k)]
        if unknown or not_conn:
            raise SystemExit(f"bad connector keys: {(unknown or not_conn)[:5]}")
        with torch.no_grad():
            for k, v in sd.items():
                if params[k].shape != v.shape:
                    raise SystemExit(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(params[k].shape)}")
                params[k].copy_(v.to(params[k].dtype))
        conn_names = set(sd)
        print(f"loaded connector {args.connector}: {len(sd)} tensors")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base, revision=rev).save_pretrained(args.out)
    (args.out / "merge_info.json").write_text(json.dumps({
        "base": args.base, "revision": rev, "lora": str(args.lora) if args.lora else None,
        "connector": str(args.connector) if args.connector else None,
        "lora_layers": len(lora_targets), "connector_tensors": len(conn_names)}, indent=2) + "\n")
    print(f"saved {args.out}")
    if args.no_check:
        return

    del model
    merged = Qwen3VLForConditionalGeneration.from_pretrained(args.out, dtype=torch.bfloat16).state_dict()
    if set(merged) != set(base_sd):
        raise SystemExit(f"key sets differ: {sorted(set(merged) ^ set(base_sd))[:5]}")
    expect = lora_targets | conn_names
    vit_changed, unchanged_target, other_changed = [], [], []
    for k, v in merged.items():
        same = torch.equal(v, base_sd[k])
        if k in expect:
            if same:
                unchanged_target.append(k)
        elif not same:
            (vit_changed if "visual." in k and not is_connector(k) else other_changed).append(k)
    print(f"check: {len(expect)} expected changes, {len(unchanged_target)} unchanged, "
          f"ViT changed {len(vit_changed)}, other changed {len(other_changed)}")
    if vit_changed or other_changed or unchanged_target:
        raise SystemExit("CHECK FAILED: " + str((vit_changed + other_changed + unchanged_target)[:5]))
    print("CHECK OK: only the trained parts differ from the base model")


if __name__ == "__main__":
    main()
