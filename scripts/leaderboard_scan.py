import json, glob, os, sys, collections

DOMS = ["gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench"]
rows = []
roots = [
    "/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs",
    "/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/outputs",
]
files = []
for r in roots:
    files += glob.glob(os.path.join(r, "**", "*.json"), recursive=True)

for f in sorted(files):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict) or "results" not in d:
        continue
    res = d["results"]
    if not isinstance(res, list) or not res or not isinstance(res[0], dict):
        continue
    if "mean_acceptance" not in res[0]:
        continue
    per_dom_prompt = collections.defaultdict(list)
    per_dom_pool = collections.defaultdict(lambda: [0, 0])
    modes = set()
    for r in res:
        ds = r.get("dataset")
        modes.add(r.get("mode"))
        per_dom_prompt[ds].append(r["mean_acceptance"])
        per_dom_pool[ds][0] += r.get("accepted_sum", 0)
        per_dom_pool[ds][1] += r.get("num_blocks", 0)
    doms = [x for x in DOMS if x in per_dom_prompt] + [
        x for x in per_dom_prompt if x not in DOMS
    ]
    A = {x: sum(per_dom_prompt[x]) / len(per_dom_prompt[x]) for x in doms}
    B = {
        x: (per_dom_pool[x][0] / per_dom_pool[x][1] if per_dom_pool[x][1] else float("nan"))
        for x in doms
    }
    macroA = sum(A.values()) / len(A)
    macroB = sum(B.values()) / len(B)
    a = d.get("arguments", {})
    rows.append(
        dict(
            file=os.path.relpath(f, "/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu"),
            n=len(res),
            ndom=len(doms),
            modes=",".join(sorted(m for m in modes if m)),
            macroA=macroA,
            macroB=macroB,
            A=A,
            B=B,
            rho=a.get("gate_rho"),
            tau=a.get("gate_tau"),
            theta=a.get("front_theta"),
            margin=a.get("repair_margin", a.get("selector_keep_repair_margin")),
            draft=os.path.basename(str(a.get("lattice_head") or a.get("draft_model") or "")),
            seed=a.get("shuffle_seed"),
        )
    )

rows.sort(key=lambda r: -r["macroA"])
print(f"{'macroA':>8} {'macroB':>8} {'n':>4} {'d':>2} {'mode':<12} {'rho':>5} {'m':>4} {'seed':>5}  file")
for r in rows:
    if r["ndom"] < 6:
        continue
    print(
        f"{r['macroA']:8.4f} {r['macroB']:8.4f} {r['n']:4d} {r['ndom']:2d} {r['modes'][:12]:<12} "
        f"{str(r['rho']):>5} {str(r['margin']):>4} {str(r['seed']):>5}  {r['file']}"
    )
print()
print("=== per-domain (macroA convention = per-prompt mean, then domain mean) top 12 ===")
hdr = f"{'file':<58}" + "".join(f"{x[:8]:>9}" for x in DOMS) + f"{'MACRO':>9}"
print(hdr)
shown = 0
for r in rows:
    if r["ndom"] < 6:
        continue
    line = f"{os.path.basename(r['file'])[:57]:<58}"
    line += "".join(f"{r['A'].get(x, float('nan')):9.3f}" for x in DOMS)
    line += f"{r['macroA']:9.4f}"
    print(line)
    shown += 1
    if shown >= 14:
        break
