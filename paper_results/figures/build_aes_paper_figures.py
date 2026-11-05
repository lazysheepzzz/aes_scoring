#!/usr/bin/env python3
"""Build editable, vector AES paper figures without models, attacks or API calls.

Aggregate charts use an explicitly user-reported snapshot, NOT locally audited
raw results. The optional search-budget plot requires compatible saved histories.
No history is inferred from final scores. Old figures/checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).with_name("seed42_reported_results.json")
DEFAULT_OUTPUT = ROOT / "outputs/paper_figures/seed42_run01"
RHI = ("rudimentary", "hotflip", "injection_family")
ATTACK_NAMES = ("Rudimentary", "HotFlip", "Injection")
SUBTYPES = ("injection_external", "injection_self_dup")
FORMATS = ("pdf", "svg", "png")
BLUE, ORANGE, INK, MUTED = "#24608A", "#C7782D", "#223346", "#5F6C79"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_path(value):
    return str(value).replace("\\", "/").rstrip("/").casefold()


def family_values(values):
    """Keep Injection as ONE family; never average four subattacks equally."""
    injection = values.get("injection_family")
    if injection is None:
        injection = (values[SUBTYPES[0]] + values[SUBTYPES[1]]) / 2
    return [values["rudimentary"], values["hotflip"], injection]


def validate_data(data):
    if data.get("schema") != "aes_paper_figures_reported_aggregates_v1":
        raise ValueError("Unsupported figure snapshot")
    if data.get("source_status") != "USER_REPORTED_AGGREGATES_NOT_RAW_ARTIFACT_VERIFIED":
        raise ValueError("Do not silently promote reported aggregates to verified measurements")
    ids = [row["id"] for row in data["models"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate models")
    for row in data["models"]:
        for attack in (*RHI, "mlm_guided"):
            if not 0 <= row["asr"][attack] <= 1:
                raise ValueError("ASR outside [0,1]")
        macro = sum(row["asr"][a] for a in RHI) / 3
        if abs(macro - row["rhi_macro_asr"]) > 1e-4:
            raise ValueError(f"Incorrect equal-family macro: {row['id']}")
    by_id = {row["id"]: row for row in data["models"]}
    if not by_id["mixed"]["checkpoint_suffix"].endswith("/gstep1000"):
        raise ValueError("This snapshot requires the uniform-30 Mixed-AT checkpoint")
    if not by_id["paer"]["checkpoint_suffix"].endswith("/gstep400"):
        raise ValueError("Unexpected PAER checkpoint")


def first_hit_curve(payload, *, n, threshold, max_rounds):
    """First threshold crossing in recorded SEARCH rounds, not accepted-edit count.

    Failed searches never become successes at their ceiling. Already successful
    searches remain counted for all later ceilings. This does not reconstruct
    scores after early stopping or claim equal query cost across attack families.
    """
    rows, summary = payload["details"], payload["summary"]
    if (len(rows) != n or float(summary["success_threshold"]) != threshold
            or not math.isfinite(float(summary["asr"]))):
        raise ValueError("History sample count/threshold mismatch")
    ids = [str(row["essay_id"]) for row in rows]
    if len(set(ids)) != n:
        raise ValueError("Duplicate essay identities")
    hits = []
    for row in rows:
        before, after = float(row["original_score"]), float(row["perturbed_score"])
        if not all(math.isfinite(x) for x in (before, after)):
            raise ValueError("Nonfinite final score")
        success = after - before >= threshold
        if not isinstance(row["success"], bool) or row["success"] != success:
            raise ValueError("Stored success differs from final scores")
        history = row.get("history")
        if not isinstance(history, list):
            raise ValueError("Missing history; cannot infer it from the final ASR")
        previous_step, first = -1, None
        for entry in history:
            step = entry["step"]
            if (not isinstance(step, int) or isinstance(step, bool)
                    or not previous_step < step < max_rounds):
                raise ValueError("Expected strictly increasing zero-based search-round indices")
            previous_step = step
            score = float(entry["score"])
            delta = float(entry["delta"])
            if not math.isfinite(score) or not math.isfinite(delta):
                raise ValueError("Nonfinite history score")
            if abs(score - before - delta) > 1e-4:
                raise ValueError("History delta differs from recorded score")
            if first is None and delta >= threshold:
                first = step + 1
        if success != (first is not None):
            raise ValueError("Final success cannot be reconciled with saved first-crossing history")
        if history and abs(float(history[-1]["score"]) - after) > 1e-4:
            raise ValueError("Saved history does not end at the returned attack text score")
        hits.append(first)
    curve = [sum(hit is not None and hit <= t for hit in hits) / n
             for t in range(max_rounds + 1)]
    if abs(curve[-1] - float(summary["asr"])) > 1e-4:
        raise ValueError("First-hit endpoint differs from the saved ASR summary")
    identities = [(str(row["essay_id"]), row["original_text"]) for row in rows]
    return curve, identities


def collect_histories(results_root, data):
    """Read-only discovery. Require checkpoint, sample, seed and search agreement.

    Different valid contents for the same model/attack are ambiguous, not an
    invitation to choose a favorable curve. Aggregate figures are never replaced
    by these discovered data. No raw-weight/data hash audit is claimed here.
    """
    n, threshold, rounds = data["n_essays"], data["success_threshold"], data["max_search_rounds"]
    settings = data["attack_settings"]
    wanted = {m["id"]: m for m in data["models"] if m["id"] in ("b0", "c0", "mixed", "paer")}
    candidates, rejected = {}, []
    for manifest_path in sorted(Path(results_root).rglob("run_manifest.json")):
        try:
            manifest = read_json(manifest_path)
            victim = normalize_path(manifest.get("victim", ""))
            matched = [m for m in wanted.values()
                       if victim.endswith(normalize_path(m["checkpoint_suffix"]))]
            if not matched or manifest.get("attack") not in (*RHI, *SUBTYPES):
                continue
            model = matched[0]
            params = manifest.get("attack_parameters", {})
            if (manifest.get("n_essays") != n or manifest.get("seed") != data["attack_seed"]
                    or params.get("n_steps") != rounds or params.get("beam_size") != settings["beam_size"]
                    or params.get("max_candidates_per_step") != settings["max_candidates_per_step"]
                    or float(params.get("success_threshold", -1)) != threshold
                    or float(manifest.get("success_threshold", -1)) != threshold
                    or normalize_path(manifest.get("data", "")).split("/")[-1] != "valid_fold0.csv"):
                raise ValueError("Manifest is not the expected full-fold seed42 greedy-30 protocol")
            if manifest["attack"] in ("rudimentary", "hotflip"):
                if params.get("max_token_edit_rate") != settings["max_token_edit_rate"]:
                    raise ValueError("Token-edit-rate cap differs from the reference evaluation")
            if manifest["attack"] == "hotflip":
                if (params.get("n_sample_pos") != settings["hotflip_n_sample_pos"]
                        or params.get("top_k_per_pos") != settings["hotflip_top_k_per_pos"]):
                    raise ValueError("HotFlip candidate proposal settings differ")
            if manifest["attack"] in ("injection_family", "injection_external"):
                if params.get("sentence_bank_sha256") != settings["sentence_bank_sha256"]:
                    raise ValueError("External Injection sentence bank differs")
            clean = read_json(manifest_path.parent / "clean_qwk.json")
            if clean.get("n") != n or abs(float(clean["qwk"]) - model["clean_qwk"]) > 1e-4:
                raise ValueError("Clean metrics do not match the intended checkpoint evaluation")
            attacks = SUBTYPES if manifest["attack"] == "injection_family" else (manifest["attack"],)
            for attack in attacks:
                detail_path = manifest_path.parent / f"{attack}_details.json"
                payload = read_json(detail_path)
                if payload["summary"]["attack"] != attack:
                    raise ValueError("Attack identity differs from detail summary")
                curve, identities = first_hit_curve(payload, n=n, threshold=threshold, max_rounds=rounds)
                if attack in RHI and abs(curve[-1] - model["asr"][attack]) > 1e-4:
                    raise ValueError("History endpoint differs from reported uniform-30 result")
                key = (model["id"], attack)
                record = {"curve": curve, "identities": identities, "details": str(detail_path.resolve()),
                          "manifest": str(manifest_path.resolve()), "sha256": digest(detail_path),
                          "manifest_sha256": digest(manifest_path)}
                existing = candidates.setdefault(key, {})
                existing.setdefault(record["sha256"], record)
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append({"manifest": str(manifest_path), "reason": str(error)})
    curves, sources, pending, common_ids = {}, [], [], None
    for model_id in wanted:
        selected = {}
        for attack in ("rudimentary", "hotflip", *SUBTYPES):
            versions = candidates.get((model_id, attack), {})
            if len(versions) != 1:
                pending.append({"model": model_id, "attack": attack,
                                "reason": "not found" if not versions else "ambiguous differing detail files"})
                continue
            selected[attack] = next(iter(versions.values()))
        if len(selected) != 4:
            continue
        identities = selected["rudimentary"]["identities"]
        if any(record["identities"] != identities for record in selected.values()):
            pending.append({"model": model_id, "reason": "Different essay identities/order across attacks"})
            continue
        if common_ids is not None and identities != common_ids:
            pending.append({"model": model_id, "reason": "Different essay identities/order across models"})
            continue
        injection = [(a + b) / 2 for a, b in zip(selected[SUBTYPES[0]]["curve"], selected[SUBTYPES[1]]["curve"])]
        if abs(injection[-1] - wanted[model_id]["asr"]["injection_family"]) > 2e-4:
            pending.append({"model": model_id, "reason": "Injection mean differs from intended result"})
            continue
        common_ids = identities
        curves[model_id] = {"rudimentary": selected["rudimentary"]["curve"],
                            "hotflip": selected["hotflip"]["curve"], "injection_family": injection}
        sources.extend({"model": model_id, "attack": attack,
                        **{k: record[k] for k in ("details", "manifest", "sha256", "manifest_sha256")}}
                       for attack, record in selected.items())
    return {"curves": curves, "sources": sources, "pending": pending, "rejected": rejected,
            "ready": "mixed" in curves and "paer" in curves,
            "note": "Metadata/history consistency checks only; not an audit of original checkpoint/data hashes."}


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#B0B9C2", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    return plt


def export(fig, output, stem, *, dpi=230):
    names = []
    for suffix in FORMATS:
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05,
                    metadata={"Creator": "AES paper figure builder"} if suffix == "pdf" else None)
        names.append(path.name)
    return names


def draw_framework(plt, output):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")

    def box(x, y, w, h, text, *, color="#F3F6F9", size=9, edge="#C6D1DC", bold=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.009",
                               linewidth=0.8, facecolor=color, edgecolor=edge)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def arrow(a, b, *, color=BLUE, dashed=False, connection="arc3"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
                                    color=color, linestyle="--" if dashed else "-",
                                    connectionstyle=connection, shrinkA=2, shrinkB=2))

    ax.text(.025, .975, "A  Offline attack-derived supervision", fontsize=11, weight="bold", va="top")
    box(.03, .795, .14, .105, "Training essays\n(original scores)")
    box(.215, .78, .34, .135,
        "Frozen B0 scorer + greedy R/H/I search\nR: character / word edits   H: token swaps\nI: external sentence / self-duplication", size=8.4)
    box(.60, .78, .365, .135,
        "Saved states and weak supervision\nToken alignment: edited-position targets\nB0 cumulative gain: correction targets", color="#FFF5E9", size=8.7)
    arrow((.17,.845),(.215,.845)); arrow((.555,.845),(.60,.845))
    ax.text(.03, .744, "MLM-guided is excluded from the offline training pool.", fontsize=8.2, color=MUTED)

    ax.text(.025, .70, "B  Trainable PAER-v3 scorer", fontsize=11, weight="bold", va="top")
    box(.03, .395, .14, .13, "Current essay\n(clean or attacked)")
    box(.215, .395, .16, .13, "Shared DeBERTa\nencoder", color="#EAF2F8", bold=True)
    arrow((.17,.46),(.215,.46))
    ax.text(.39,.46,r"$h_i$", fontsize=11, ha="center", va="center")
    box(.435,.60,.18,.075,"Global regression head", size=8.4)
    box(.435,.48,.18,.065,r"Attention head  $\alpha_i$", size=9)
    box(.435,.375,.18,.065,r"Risk head  $r_i$", size=9)
    box(.435,.27,.18,.065,r"Signed evidence  $e_i$", size=9)
    for yy in (.5125,.4075,.3025):
        arrow((.407,.46),(.435,yy))
    arrow((.375,.50),(.435,.6375), connection="angle,angleA=90,angleB=180,rad=8")
    box(.68,.29,.285,.255,"", color="#EDF5F1", edge="#B9D0C5")
    ax.text(.8225,.51,"Directional aggregation",ha="center",weight="bold",fontsize=9.2)
    ax.text(.8225,.446,r"$e_i^+=\mathrm{ReLU}(e_i),\quad e_i^-=\mathrm{ReLU}(-e_i)$",ha="center",fontsize=10)
    ax.text(.8225,.38,r"$\ell(x)=\sum_i\alpha_i[(1-\lambda_r r_i)e_i^+-e_i^-]$",ha="center",fontsize=10.4)
    ax.text(.8225,.322,"Negative local evidence is not risk-gated",ha="center",fontsize=7.5,color=MUTED)
    for start, end in zip((.5125,.4075,.3025),(.49,.42,.345)):
        arrow((.615,start),(.68,end))
    box(.77,.59,.195,.095,"Final score\n"+r"$f(x)=g(x)+\ell(x)$",color="#EAF2F8",size=10)
    arrow((.615,.6375),(.77,.6375))
    ax.text(.69,.66,r"$g(x)$",ha="center",fontsize=10,color=BLUE)
    arrow((.865,.545),(.865,.59))
    ax.text(.03,.245,"The global score is jointly trained but is not directly risk-gated.",fontsize=8.3,color=MUTED)

    ax.plot([.025,.975],[.22,.22],color="#CCD5DD",lw=.8)
    box(.03,.045,.57,.135,"TRAINING ONLY\nOriginal-score regression + one-sided inflation control\nLocalization / attention + correction calibration / lift + clean regularization",
        color="#FFF5E9",size=8.3)
    box(.635,.045,.33,.135,"INFERENCE\nOne essay in, one score out\nNo trajectory, paired original, or attack label",color="#EAF2F8",size=8.7)
    arrow((.785,.775),(.785,.716),color=ORANGE,dashed=True)
    ax.text(.79,.718,"targets supervise training objectives",va="center",fontsize=7.5,color="#9A5B20")
    names = export(fig, output, "aes_paer_framework")
    plt.close(fig)
    return names


def draw_heatmap(plt, output, data):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    rows = data["models"]
    matrix = np.array([[m["asr"][a]*100 for a in (*RHI,"mlm_guided")] for m in rows])
    fig, (ax, qax) = plt.subplots(1,2,figsize=(8.4,4.5),gridspec_kw={"width_ratios":[5.4,1.1],"wspace":.20})
    fig.subplots_adjust(left=.275,right=.94,top=.84,bottom=.20)
    cmap = LinearSegmentedColormap.from_list("inflation", ["#F1F6F8","#AEC5D5","#426985","#173B59"])
    image = ax.imshow(matrix,vmin=0,vmax=100,cmap=cmap,aspect="auto")
    ax.set_xticks(range(4),["Rudimentary","HotFlip","Injection","MLM-guided"],fontsize=8.2)
    ax.set_yticks(range(len(rows)),[m["label"] for m in rows],fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="minor",length=0)
    ax.axvline(2.5,color="#D18A47",ls=(0,(3,2)),lw=1.5)
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",fontsize=9.1,
                    color="white" if matrix[i,j]>58 else INK,
                    weight="bold" if rows[i]["id"]=="paer" else "normal")
    ax.add_patch(Rectangle((-.5,len(rows)-1.5),4,1,fill=False,edgecolor=ORANGE,lw=1.5))
    for spine in ax.spines.values():spine.set_visible(False)
    qwk = [m["clean_qwk"] for m in rows]
    colors = [ORANGE if m["id"]=="paer" else BLUE for m in rows]
    qax.scatter(qwk,range(len(rows)),c=colors,s=30,zorder=3)
    qax.set_ylim(len(rows)-.5,-.5);qax.set_xlim(.812,.850)
    qax.set_yticks([]);qax.set_xticks([.82,.84]);qax.tick_params(axis="x",labelsize=8)
    qax.set_title("Clean QWK ↑",fontsize=9,pad=12)
    qax.grid(axis="x",color="#E5E9ED",lw=.7)
    for i,q in enumerate(qwk): qax.text(q+.0014,i,f"{q:.4f}",va="center",fontsize=7.5)
    cax=fig.add_axes([.29,.078,.43,.025]);fig.colorbar(image,cax=cax,orientation="horizontal",label="Attack success rate (%) ↓")
    fig.text(.465,.925,"R/H/I attack families",ha="center",fontsize=9,weight="bold")
    fig.text(.712,.925,"MLM transfer",ha="center",fontsize=8,color="#9A5B20")
    names=export(fig,output,"aes_cross_attack_heatmap")
    plt.close(fig)
    return names


def draw_routing(plt, output, data):
    import numpy as np
    fixed, adaptive = (data["routing_diagnostic"][k] for k in ("fixed_set","adaptive"))
    fig, axes=plt.subplots(1,2,figsize=(8.3,3.7),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,top=.78,bottom=.30,wspace=.13)
    x=np.arange(3);w=.33
    for ax,mode,title in zip(axes,(fixed,adaptive),("(a) Same saved attack texts","(b) Each mode attacked separately")):
        off,on=([v*100 for v in family_values(mode[k])] for k in ("route_off","routed"))
        bars0=ax.bar(x-w/2,off,w,color=BLUE,label="Routing off",zorder=3)
        bars1=ax.bar(x+w/2,on,w,color=ORANGE,label="Routing on",zorder=3)
        ax.bar_label(bars0,labels=[f"{v:.1f}" for v in off],fontsize=8,padding=3)
        ax.bar_label(bars1,labels=[f"{v:.1f}" for v in on],fontsize=8,padding=3)
        ax.set_xticks(x,ATTACK_NAMES,fontsize=8.7);ax.set_ylim(0,110)
        ax.set_title(title,loc="left",fontsize=9.4,pad=16,weight="bold")
        ax.set_yticks([0,25,50,75,100]);ax.grid(axis="y",color="#E6EBEF",zorder=0)
        delta=(sum(on)-sum(off))/3
        center=(ax.get_position().x0+ax.get_position().x1)/2
        fig.text(center,.065,f"RHI macro: off {sum(off)/3:.2f}% → on {sum(on)/3:.2f}%\nOn − off: {delta:+.2f} percentage points",
                 ha="center",fontsize=8.5,color=MUTED,linespacing=1.5)
    axes[0].set_ylabel("Attack success rate (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.99),ncol=2,frameon=False)
    names=export(fig,output,"aes_routing_diagnostics")
    plt.close(fig)
    return names


def draw_budget(plt, output, data, histories):
    fig,axes=plt.subplots(1,3,figsize=(8.4,2.9),sharey=True)
    fig.subplots_adjust(left=.075,right=.99,top=.78,bottom=.20,wspace=.12)
    styles={"b0":("#8996A3","--"),"c0":("#A286A5",":"),"mixed":(BLUE,"-"),"paer":(ORANGE,"-")}
    models={m["id"]:m for m in data["models"]}
    t=list(range(data["max_search_rounds"]+1))
    for ax,attack,title in zip(axes,RHI,ATTACK_NAMES):
        for model_id,curves in histories["curves"].items():
            color,ls=styles[model_id]
            ax.step(t,[100*v for v in curves[attack]],where="post",color=color,ls=ls,lw=1.8,label=models[model_id]["short_label"])
        ax.set_title(title,weight="bold");ax.set_xlim(0,30);ax.set_ylim(0,102)
        ax.set_xticks([0,10,20,30]);ax.set_xlabel("Search rounds")
        ax.grid(color="#E7ECF0",lw=.7)
    axes[0].set_ylabel("Cumulative attack success (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=4,frameon=False)
    names=export(fig,output,"aes_attack_budget_curves")
    plt.close(fig)
    return names


def guard_output(output, data_path):
    output=Path(output).resolve()
    for protected in (ROOT/"data",ROOT/"deberta_checkpoints",data_path.parent):
        if output.is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output):
            raise ValueError("Output must not overlap protected inputs/source directory")
    marker=output/"figure_manifest.json"
    if marker.exists():
        raise FileExistsError("Figure set already exists; use a NEW --output-dir to preserve it")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty output directory")
    return output


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--results-root",type=Path,help="Read-only scan of remote outputs for real compatible attack histories")
    p.add_argument("--dry-run",action="store_true",help="Validate/report only; no images/files, GPU, network or training")
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    data=read_json(args.data);validate_data(data)
    output=guard_output(args.output_dir,args.data.resolve())
    history={"ready":False,"curves":{},"sources":[],"pending":[{"reason":"No remote --results-root supplied"}],"rejected":[]}
    if args.results_root:
        if not args.results_root.is_dir():raise FileNotFoundError(args.results_root)
        history=collect_histories(args.results_root,data)
    status={"mode":"OFFLINE_PLOTTING_NO_GPU_NO_API_NO_TRAINING",
            "data_status":data["source_status"],"output_dir":str(output),
            "framework":"ready","heatmap":"reported-aggregate draft",
            "routing_diagnostic":"reported-aggregate draft",
            "search_budget":"ready from saved histories" if history["ready"] else "pending; no plot will be fabricated",
            "history_models_available":list(history["curves"]),"history_pending":history["pending"]}
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if args.dry_run:return 0
    output.mkdir(parents=True)
    plt=plotting()
    files=draw_framework(plt,output)+draw_heatmap(plt,output,data)+draw_routing(plt,output,data)
    if history["ready"]:files+=draw_budget(plt,output,data,history)
    save_json(output/"search_budget_data_status.json",history)
    save_json(output/"reported_aggregate_source.json",data)
    save_json(output/"figure_manifest.json",{**status,"data_sha256":digest(args.data),
              "builder_sha256":digest(Path(__file__)),"files":{name:digest(output/name) for name in files},
              "training_seeds":1,"uncertainty_bands":"not drawn; no invented multiseed error bars",
              "history_sources":history["sources"]})
    print(f"[SAVED] {output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
