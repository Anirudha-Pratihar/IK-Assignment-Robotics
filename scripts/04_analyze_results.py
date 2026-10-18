#!/usr/bin/env python3
"""STEP 4 - metrics tables, paired statistics, failure breakdown and plots for the report.

  python scripts/04_analyze_results.py --tag main
  python scripts/04_analyze_results.py --tag stress --extra main,stress_retry4,tray_noCA     # + cross-experiment success table
Writes: results/<tag>/summary.csv|md, report/figures/<tag>_*.png, report/tables/<tag>_*.tex  (\\input{} them in report/main.tex)
Because every method ran the SAME seeds the comparison is PAIRED: McNemar (exact) for success, Wilcoxon signed-rank for continuous metrics."""
import os, sys, json, argparse, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from pick_place.metrics import load_jsonl, FAIL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG, TAB = os.path.join(ROOT, "report", "figures"), os.path.join(ROOT, "report", "tables")
RES = os.path.join(ROOT, "results")            # set to results/franka for --robot franka
COL = {"mink": "#4c72b0", "dls": "#dd8452", "qp": "#55a868"}
NAME = {"mink": "Mink", "dls": "DLS", "qp": "QP"}


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(tag):
    d = os.path.join(RES, tag); rows = []
    for m in ("mink", "dls", "qp"):
        p = os.path.join(d, f"episodes_{m}.jsonl")
        if os.path.exists(p): rows += load_jsonl(p)
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit(f"no logs in {d}")
    df["method"] = pd.Categorical(df["method"], [m for m in NAME if m in set(df["method"])])
    return df, d


def tex_escape(s): return str(s).replace("_", "\\_").replace("%", "\\%")


def summary(df):
    out = []
    for m, g in df.groupby("method", observed=True):
        n, k = len(g), int(g.success.sum()); lo, hi = wilson(k, n); s = g[g.success]
        out.append(dict(method=NAME[m], n=n, success=k, success_pct=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi,
                        plan_ms=1e3 * g.plan_time_s.mean(), plan_ms_sd=1e3 * g.plan_time_s.std(),
                        ik_iters=g.ik_iters.mean(), ik_iters_sd=g.ik_iters.std(), conv_fail=int(g.conv_fail.sum()),
                        clr_robot_mm=1e3 * g.min_clear_robot_env.min(), clr_robot_mean_mm=1e3 * g.min_clear_robot_env.mean(),
                        clr_cube_mm=1e3 * g.min_clear_cube_tray.min(), max_dq=g.max_joint_vel.max(),
                        vel_ev=int(g.vel_events.sum()), lim_ev=int(g.limit_events.sum()), qp_fail=int(g.qp_fail.sum()),
                        track_mm=g.track_rms_mm.mean(), place_mm=s.place_err_mm.mean() if len(s) else np.nan,
                        exec_s=s.exec_time_s.mean() if len(s) else np.nan))
    return pd.DataFrame(out)


def paired_tests(df):
    """Seeds shared by both methods -> PAIRED tests (exact McNemar, Wilcoxon signed-rank).
       Fewer than 8 shared seeds (e.g. the combined --split run) -> UNPAIRED tests (Fisher exact, Mann-Whitney U)."""
    ms = list(df.method.cat.categories); rows = []
    for a, b in itertools.combinations(ms, 2):
        A, B = df[df.method == a].set_index("episode"), df[df.method == b].set_index("episode")
        common = sorted(set(A.index) & set(B.index))
        if len(common) >= 8:
            Sa, Sb = A.loc[common, "success"].astype(int), B.loc[common, "success"].astype(int)
            b01, b10 = int(((Sa == 1) & (Sb == 0)).sum()), int(((Sa == 0) & (Sb == 1)).sum())
            p_s = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
            def cont(c):
                d = (A.loc[common, c] - B.loc[common, c]).dropna()
                return 1.0 if (len(d) < 3 or (d == 0).all()) else stats.wilcoxon(d).pvalue
            rows.append(dict(pair=f"{NAME[a]} vs {NAME[b]}", test="paired", n_a=len(common), n_b=len(common), only_first=b01, only_second=b10,
                             p_success=p_s, p_time=cont("plan_time_s"), p_iters=cont("ik_iters")))
        else:
            ka, kb = int(A.success.sum()), int(B.success.sum())
            p_s = stats.fisher_exact([[ka, len(A) - ka], [kb, len(B) - kb]])[1]
            def mw(c):
                x, y = A[c].dropna().values, B[c].dropna().values
                return 1.0 if (len(x) < 2 or len(y) < 2 or (np.ptp(np.r_[x, y]) == 0)) else stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
            rows.append(dict(pair=f"{NAME[a]} vs {NAME[b]}", test="unpaired", n_a=len(A), n_b=len(B), only_first=np.nan, only_second=np.nan,
                             p_success=p_s, p_time=mw("plan_time_s"), p_iters=mw("ik_iters")))
    return pd.DataFrame(rows)


def coverage(df, target):
    """episodes done per method vs the target (e.g. 25 each) and the seeds still missing"""
    rows = []
    for m in df.method.cat.categories:
        done = set(df[df.method == m].episode); missing = [e for e in range(target) if e not in done]
        rows.append(dict(method=NAME[m], done=len(done), target=target, remaining=len(missing), missing_seeds=missing))
    return pd.DataFrame(rows)


def fmt_p(p): return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def write_tex(tag, S, F, P):
    os.makedirs(TAB, exist_ok=True)
    L = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
         "Method & Succ. & Rate [\\%] (95\\% CI) & Solve [ms] & IK it. & Min clr [mm] & Cube--tray [mm] & Vel.\\ viol. & Limit ev. & Place err [mm]\\\\", "\\midrule"]
    for _, r in S.iterrows():
        L.append(f"{r.method} & {r.success}/{r.n} & {r.success_pct:.0f} ({r.ci_lo:.0f}--{r.ci_hi:.0f}) & {r.plan_ms:.0f}$\\pm${r.plan_ms_sd:.0f} & "
                 f"{r.ik_iters:.0f}$\\pm${r.ik_iters_sd:.0f} & {r.clr_robot_mm:.0f} & {r.clr_cube_mm:.0f} & {r.vel_ev} & {r.lim_ev} & {r.place_mm:.1f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open(f"{TAB}/{tag}_summary.tex", "w").write("\n".join(L) + "\n")
    cols = [c for c in FAIL if c in F.columns] or list(F.columns)
    L = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "Method & " + " & ".join(tex_escape(c) for c in cols) + "\\\\", "\\midrule"]
    for m, r in F.iterrows(): L.append(f"{m} & " + " & ".join(str(int(r[c])) for c in cols) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open(f"{TAB}/{tag}_failures.tex", "w").write("\n".join(L) + "\n")
    if (P.test == "paired").all():
        L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Pair & only 1st ok & only 2nd ok & $p$ success & $p$ solve time & $p$ IK it.\\\\", "\\midrule"]
        for _, r in P.iterrows(): L.append(f"{r.pair} & {int(r.only_first)} & {int(r.only_second)} & {fmt_p(r.p_success)} & {fmt_p(r.p_time)} & {fmt_p(r.p_iters)}\\\\")
    else:
        L = ["\\begin{tabular}{lllrrr}", "\\toprule", "Pair & test & $n$ (1st/2nd) & $p$ success & $p$ solve time & $p$ IK it.\\\\", "\\midrule"]
        for _, r in P.iterrows(): L.append(f"{r.pair} & {r.test} & {int(r.n_a)}/{int(r.n_b)} & {fmt_p(r.p_success)} & {fmt_p(r.p_time)} & {fmt_p(r.p_iters)}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open(f"{TAB}/{tag}_tests.tex", "w").write("\n".join(L) + "\n")


def plots(tag, df, design):
    os.makedirs(FIG, exist_ok=True); ms = list(df.method.cat.categories)
    S = summary(df)
    # success bars with Wilson CI
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for i, r in S.reset_index().iterrows():
        m = ms[i]; ax.bar(i, r.success_pct, color=COL[m]); ax.errorbar(i, r.success_pct, yerr=[[r.success_pct - r.ci_lo], [r.ci_hi - r.success_pct]], c="k", capsize=4)
        ax.text(i, 3, f"{int(r.success)}/{int(r.n)}", ha="center", color="w", fontsize=9)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAME[m] for m in ms]); ax.set_ylabel("success rate [%] (95% Wilson CI)"); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_success.png", dpi=180); plt.close(fig)
    # boxplots
    spec = [("plan_time_s", "solve time / episode [ms]", 1e3), ("ik_iters", "IK iterations / episode", 1), ("min_clear_robot_env", "min robot-env clearance [mm]", 1e3),
            ("min_clear_cube_tray", "min cube-tray clearance [mm]", 1e3), ("max_joint_vel", "max |dq| commanded [rad/s]", 1), ("track_rms_mm", "TCP tracking RMS [mm]", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, (c, lab, sc) in zip(axs.ravel(), spec):
        data = [df[df.method == m][c].dropna().values * sc for m in ms]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="k"))
        for patch, m in zip(bp["boxes"], ms): patch.set_facecolor(COL[m])
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([NAME[m] for m in ms]); ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_metrics.png", dpi=180); plt.close(fig)
    # failure stack
    F = pd.crosstab(df[~df.success].method, df[~df.success].failure_reason).reindex(ms).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 3.2)); bottom = np.zeros(len(ms))
    for rsn in F.columns:
        ax.bar([NAME[m] for m in ms], F[rsn].values, bottom=bottom, label=rsn); bottom += F[rsn].values
    ax.set_ylabel("failed episodes"); 
    if len(F.columns): ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_failures.png", dpi=180); plt.close(fig)
    # cube map
    fig, axs = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, ms):
        g = df[df.method == m]
        ax.add_patch(Rectangle((design["ymin"], design["xmin"]), design["W"], design["L"], fill=False, ec="k"))
        ty, tx = design["tray_center"][1], design["tray_center"][0]; ax.add_patch(Rectangle((ty - 0.093, tx - 0.093), 0.186, 0.186, fc="#444", alpha=.6))
        ax.scatter(g[g.success].cube_y, g[g.success].cube_x, c="#2ca02c", s=22, label="success")
        ax.scatter(g[~g.success].cube_y, g[~g.success].cube_x, c="#d62728", marker="x", s=40, label="fail")
        ax.set_title(NAME[m]); ax.set_xlabel("y [m]"); ax.invert_xaxis(); ax.set_aspect("equal"); ax.plot(design.get("base_xy", [0, 0])[1], design.get("base_xy", [0, 0])[0], "r^")
    axs[0].set_ylabel("x [m]"); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_cubemap.png", dpi=180); plt.close(fig)
    # iterations by phase
    ph = ["approach", "descend", "lift", "transit", "place", "retreat"]
    fig, ax = plt.subplots(figsize=(6, 3.2)); w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        g = df[df.method == m]; vals = [np.nanmean([d.get(p, np.nan) if isinstance(d, dict) else np.nan for d in g.ik_iters_by_phase]) for p in ph]
        ax.bar(np.arange(len(ph)) + i * w, vals, w, color=COL[m], label=NAME[m])
    ax.set_xticks(np.arange(len(ph)) + w * (len(ms) - 1) / 2); ax.set_xticklabels(ph); ax.set_ylabel("IK iterations to converge"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_iters_phase.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--extra", default="")
    ap.add_argument("--robot", default="heal", choices=["heal", "franka"]); ap.add_argument("--target", type=int, default=25, help="episodes per method that the full study needs (for the coverage report)")
    a = ap.parse_args()
    global RES
    RES = os.path.join(ROOT, "results") if a.robot == "heal" else os.path.join(ROOT, "results", a.robot)
    pre = "" if a.robot == "heal" else f"{a.robot}_"                       # HEAL output names stay unchanged
    df, d = load(a.tag); design = json.load(open(os.path.join(d, "design.json")))
    S = summary(df); P = paired_tests(df)
    F = pd.crosstab(df[~df.success].method, df[~df.success].failure_reason).reindex(list(df.method.cat.categories)).fillna(0)
    F.index = [NAME[m] for m in F.index]
    S.to_csv(os.path.join(d, "summary.csv"), index=False)
    write_tex(pre + a.tag, S, F, P); plots(pre + a.tag, df, design)
    C = coverage(df, a.target); C.to_csv(os.path.join(d, "coverage.csv"), index=False)
    seeds_ok = len(set(map(tuple, df.groupby('method', observed=True).episode.apply(lambda x: tuple(sorted(x))).tolist()))) == 1
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(f"# {a.tag}\n\n" + S.round(2).to_markdown(index=False) + "\n\n## failures\n\n" + (F.to_markdown() if len(F) else "none") + "\n\n## tests\n\n" + P.round(4).to_markdown(index=False) + "\n\n## coverage\n\n" + C.drop(columns="missing_seeds").to_markdown(index=False) + "\n")
    print(S.round(2).to_string(index=False)); print("\nfailures by reason:\n", F.to_string() if len(F) else "none"); print("\ntests:\n", P.round(4).to_string(index=False))
    print("\ncoverage (episodes done per method vs target %d):\n" % a.target, C.drop(columns="missing_seeds").to_string(index=False), "\nremaining in total:", int(C.remaining.sum()), "| every method ran the same seeds:", seeds_ok)
    print("episode/seed overlap between methods:", {a_: {b_: len(set(df[df.method == a_].episode) & set(df[df.method == b_].episode)) for b_ in df.method.cat.categories if b_ != a_} for a_ in df.method.cat.categories})
    # per-failure list (for the failure-analysis section / screenshots)
    bad = df[~df.success][["method", "episode", "seed", "cube_x", "cube_y", "cube_yaw", "failure_reason", "failure_phase", "failure_detail"]]
    bad.to_csv(os.path.join(d, "failures.csv"), index=False)
    if a.extra:
        rows = []
        for t in [a.tag] + [x for x in a.extra.split(",") if x and x != a.tag]:
            try: e, _ = load(t)
            except SystemExit: continue
            for m, g in e.groupby("method", observed=True): rows.append(dict(experiment=t, method=NAME[m], success=f"{int(g.success.sum())}/{len(g)}", pct=100 * g.success.mean()))
        order = ["main", "stress", "stress_retry4", "lowcarry_noCA", "lowcarry_ca", "lowcarry_smart", "fast"]
        X = pd.DataFrame(rows).pivot(index="experiment", columns="method", values="success"); X = X.reindex([t for t in order if t in X.index] + [t for t in X.index if t not in order]); X = X[[c for c in ("Mink", "DLS", "QP") if c in X.columns]]; print("\ncross-experiment success:\n", X.to_string())
        L = ["\\begin{tabular}{l" + "r" * len(X.columns) + "}", "\\toprule", "Experiment & " + " & ".join(X.columns) + "\\\\", "\\midrule"]
        for t, r in X.iterrows(): L.append(tex_escape(t) + " & " + " & ".join(str(v) for v in r.values) + "\\\\")
        L += ["\\bottomrule", "\\end{tabular}"]; open(f"{TAB}/{pre}experiments.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
