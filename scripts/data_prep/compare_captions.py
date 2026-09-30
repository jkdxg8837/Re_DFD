# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen recaptions of ViPE videos with their original HF `long` captions (prompt validation, spec §2.3).

Usage: python scripts/data_prep/compare_captions.py <qwen.jsonl> [--orig_dir data/caption_val/vipe] [--out report.md]
Reports length/style statistics for both sets and the text-embedding cosine of matched pairs (same video) vs.
mismatched pairs (different videos), using sentence-transformers all-mpnet-base-v2. Run with .venv-vlm.
"""

import argparse
import json
import os
import re

import numpy as np

SENT = re.compile(r"[.!?](?:\s|$)")


def stats(caps):
    w = np.array([len(c.split()) for c in caps])
    s = np.array([max(1, len(SENT.findall(c))) for c in caps])
    first = [c.split()[0].lower() if c.split() else "" for c in caps]
    return {
        "n": len(caps),
        "words_mean": float(w.mean()), "words_median": float(np.median(w)),
        "words_p5": float(np.percentile(w, 5)), "words_p95": float(np.percentile(w, 95)),
        "words_min": int(w.min()), "words_max": int(w.max()),
        "sentences_mean": float(s.mean()),
        "starts_with_a_an": float(np.mean([f in ("a", "an") for f in first])),
        "starts_with_the_video_or_in_this": float(np.mean([bool(re.match(r"(the video|in this|this video)", c.lower())) for c in caps])),
        "has_newline_or_list": float(np.mean([("\n" in c) or bool(re.search(r"(^|\s)[-*•]\s|\d\.\s", c)) for c in caps])),
        "mentions_camera": float(np.mean(["camera" in c.lower() for c in caps])),
        # share of the most common 5-word opening (catches prompt-example leakage)
        "top_opening_share": float(max(np.unique([" ".join(c.split()[:5]).lower() for c in caps], return_counts=True)[1]) / len(caps)),
        "starts_young_woman_long_dark_hair": float(np.mean([c.startswith("A young woman with long dark hair") for c in caps])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qwen")
    ap.add_argument("--orig_dir", default="data/caption_val/vipe")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.qwen)]
    gen = [r["caption"] for r in recs]
    orig = [" ".join(open(os.path.join(a.orig_dir, r["id"] + ".txt")).read().split()) for r in recs]

    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    eg = m.encode(gen, normalize_embeddings=True)
    eo = m.encode(orig, normalize_embeddings=True)
    sim = eg @ eo.T
    matched = np.diag(sim)
    mismatched = sim[~np.eye(len(gen), dtype=bool)]
    rank1 = float(np.mean(sim.argmax(1) == np.arange(len(gen))))

    sg, so = stats(gen), stats(orig)
    lines = [f"# Caption validation: {os.path.basename(a.qwen)} ({len(gen)} ViPE videos)", "",
             "| stat | Qwen recaption | original HF `long` |", "|---|---|---|"]
    for k in sg:
        f = (lambda x: f"{x:.3f}") if isinstance(sg[k], float) else str
        lines.append(f"| {k} | {f(sg[k])} | {f(so[k])} |")
    lines += ["", f"Embedding cosine (all-mpnet-base-v2): matched mean {matched.mean():.3f} (min {matched.min():.3f}), "
              f"mismatched mean {mismatched.mean():.3f}; retrieval top-1 (Qwen -> original) {rank1:.2f}", "",
              "## Examples", ""]
    for r, o in list(zip(recs, orig))[:5]:
        lines += [f"**{r['id']}** ({r['n_words']} vs {len(o.split())} words)", "", f"- Qwen: {r['caption']}",
                  f"- orig: {o}", ""]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        json.dump({"qwen": sg, "orig": so, "matched_cos": float(matched.mean()), "mismatched_cos": float(mismatched.mean()),
                   "top1": rank1}, open(a.out.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
