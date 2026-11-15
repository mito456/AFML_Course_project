import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# ==============================================================================
# CONFIGURATION BLOCK (Fixed Constants)
# ==============================================================================
CONFIG = {
    "tau_thresholds": np.linspace(0.05, 1.0, 30),
    "bootstrap_draws": 1000,
    "bootstrap_seed": 42,
    "random_seeds": range(5),  # Seeds 0-4
    "question": "Q3",
    "k": 0,
    "tag": "[M4]"
}

print("1. Loading data files and inspecting structures...")
events_df = pd.read_csv("clip_events.csv")
summary_df = pd.read_csv("results_summary.csv")
npz_data = np.load("clip_probs.npz", allow_pickle=True)

# Dynamically load expected baseline values from M3's results_summary.csv
exp_cam_acc = summary_df.loc[summary_df["model"] == "global_visual", "accuracy"].values[0]
exp_cam_f1 = summary_df.loc[summary_df["model"] == "global_visual", "macro_f1"].values[0]
exp_aud_acc = summary_df.loc[summary_df["model"] == "audio_only", "accuracy"].values[0]
exp_aud_f1 = summary_df.loc[summary_df["model"] == "audio_only", "macro_f1"].values[0]

video_names_npz = npz_data["video_name"]
splits = npz_data["split"]
classes = npz_data["classes"]
camera_probs = npz_data["camera"]
audio_probs = npz_data["audio"]

npz_df = pd.DataFrame({
    "video_name": video_names_npz,
    "split_npz": splits,
    "npz_idx": np.arange(len(video_names_npz))
})

merged_df = pd.merge(events_df, npz_df, on="video_name", how="inner")

# ------------------------------------------------------------------------------
# SPLIT HANDLING: k=0 uses all 525 target clips (adapt + test); k>0 uses 456 test clips only
# ------------------------------------------------------------------------------
if CONFIG["k"] == 0:
    target_df = merged_df[merged_df["split_npz"].isin(["adapt", "test"])].copy()
    split_label = "target_site_525"
    shift_level = "target_site_adapt_test"
else:
    target_df = merged_df[merged_df["split_npz"] == "test"].copy()
    split_label = "test_site_456"
    shift_level = "target_site_test_only"

indices = target_df["npz_idx"].values
y_raw = target_df["species"].values
site_ids = target_df["location_id"].values

camera_target = camera_probs[indices]
audio_target = audio_probs[indices]

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
y_true = []
for val in y_raw:
    if val not in class_to_idx:
        raise ValueError(f"Label '{val}' found in target set is missing from npz classes dictionary!")
    y_true.append(class_to_idx[val])
y_true = np.array(y_true)
n_classes = len(classes)

n_clips = len(y_true)
n_sites = len(np.unique(site_ids))

# ------------------------------------------------------------------------------
# PRE-COMPUTED CSV STRING ALIGNMENT & HARD ASSERTS
# ------------------------------------------------------------------------------
vis_preds_base = np.argmax(camera_target, axis=1)
aud_preds_base = np.argmax(audio_target, axis=1)

pred_camera_strs = classes[vis_preds_base]
pred_audio_strs = classes[aud_preds_base]

if "pred_camera" in target_df.columns:
    assert np.all(pred_camera_strs == target_df["pred_camera"].values), "Mismatch between computed camera predictions and pred_camera!"
if "pred_audio" in target_df.columns:
    assert np.all(pred_audio_strs == target_df["pred_audio"].values), "Mismatch between computed audio predictions and pred_audio!"

base_cam_acc = accuracy_score(y_true, vis_preds_base)
base_cam_f1 = f1_score(y_true, vis_preds_base, average="macro", labels=range(n_classes), zero_division=0)
base_aud_acc = accuracy_score(y_true, aud_preds_base)
base_aud_f1 = f1_score(y_true, aud_preds_base, average="macro", labels=range(n_classes), zero_division=0)

print(f"\n--- M3 v1 Baseline Verification ---")
print(f"Target Clips Count: {n_clips}")
print(f"Unique Sites Count: {n_sites}")
print(f"Camera Accuracy:    {base_cam_acc:.6f} (Expected: {exp_cam_acc:.6f})")
print(f"Camera Macro-F1:    {base_cam_f1:.6f} (Expected: {exp_cam_f1:.6f})")
print(f"Audio Accuracy:     {base_aud_acc:.6f} (Expected: {exp_aud_acc:.6f})")
print(f"Audio Macro-F1:     {base_aud_f1:.6f} (Expected: {exp_aud_f1:.6f})")

assert abs(base_cam_acc - exp_cam_acc) < 0.0001, f"Camera accuracy mismatch!"
assert abs(base_cam_f1 - exp_cam_f1) < 0.0001, f"Camera macro-F1 mismatch!"
assert abs(base_aud_acc - exp_aud_acc) < 0.0001, f"Audio accuracy mismatch!"
assert abs(base_aud_f1 - exp_aud_f1) < 0.0001, f"Audio macro-F1 mismatch!"
print("All M3 v1 Hard Asserts & CSV Alignments Passed Successfully!")

# ------------------------------------------------------------------------------
# VOCAL VS. NON-VOCAL ANALYSIS (Analysis Only - using continuous ana_target_sound quartiles)
# ------------------------------------------------------------------------------
if "ana_target_sound" in target_df.columns:
    s = target_df["ana_target_sound"]
    print(f"\n--- Audio Analysis: ana_target_sound ---")
    print(f"Description:\n{s.describe()}")
    print(f"NaN count: {s.isna().sum()}")
    
    valid_mask = s.notna()
    if valid_mask.sum() > 0:
        target_df_valid = target_df[valid_mask].copy()
        target_df_valid["sound_quartile"] = pd.qcut(target_df_valid["ana_target_sound"], q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
        print("\nAudio Accuracy across ana_target_sound Quartiles:")
        for q_label, group in target_df_valid.groupby("sound_quartile"):
            g_indices = group["npz_idx"].values
            g_true_indices = [np.where(indices == idx)[0][0] for idx in g_indices]
            g_true = y_true[g_true_indices]
            g_pred = aud_preds_base[g_true_indices]
            q_acc = accuracy_score(g_true, g_pred)
            print(f"  {q_label} ({len(group)} clips) - Audio Accuracy: {q_acc:.4f}")

# ------------------------------------------------------------------------------
# TRUE OUT-OF-FOLD SOURCE FUSION (Optimized on log-probabilities using Macro-F1)
# ------------------------------------------------------------------------------
source_oof_df = merged_df[(merged_df["split_npz"] == "source") & (merged_df["cv_fold"] != -1) & (merged_df["prob_source"] == "oof_site_cv")].copy()
if len(source_oof_df) == 0:
    raise RuntimeError("Strict source out-of-fold (OOF) rows not found!")

source_indices = source_oof_df["npz_idx"].values
source_y = np.array([class_to_idx[val] for val in source_oof_df["species"].values])
source_cam_log = np.log(np.clip(camera_probs[source_indices], 1e-8, 1.0))
source_aud_log = np.log(np.clip(audio_probs[source_indices], 1e-8, 1.0))

target_cam_log = np.log(np.clip(camera_target, 1e-8, 1.0))
target_aud_log = np.log(np.clip(audio_target, 1e-8, 1.0))

# Optimize scalar fusion weight alpha on log-probabilities using Macro-F1 on source OOF
best_alpha = 0.5
best_f1 = -1.0
for alpha in np.linspace(0.0, 1.0, 101):
    blend_log = alpha * source_cam_log + (1 - alpha) * source_aud_log
    preds = np.argmax(blend_log, axis=1)
    macro_f1 = f1_score(source_y, preds, average="macro", labels=range(n_classes), zero_division=0)
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_alpha = alpha

print(f"Optimized Source OOF Log-Prob Fusion Weight (Camera Alpha): {best_alpha:.2f}")

p_fusion_log = best_alpha * target_cam_log + (1 - best_alpha) * target_aud_log
fusion_preds = np.argmax(p_fusion_log, axis=1)
vis_conf = np.max(camera_target, axis=1)
audio_preds = aud_preds_base

oracle_sensor_preds = np.copy(vis_preds_base)
for i in range(len(y_true)):
    if vis_preds_base[i] != y_true[i] and audio_preds[i] == y_true[i]:
        oracle_sensor_preds[i] = audio_preds[i]

# ------------------------------------------------------------------------------
# EVALUATION HARNESS WITH SITE-BOOTSTRAP CIs
# ------------------------------------------------------------------------------
def evaluate_policy_detailed(defer_mask, fusion_preds, vis_preds, y_true, site_array):
    final_preds = np.where(defer_mask, fusion_preds, vis_preds)
    macro_f1 = f1_score(y_true, final_preds, average="macro", labels=range(n_classes), zero_division=0)
    acc = accuracy_score(y_true, final_preds)
    act_rate = np.mean(defer_mask)
    
    unique_sites = np.unique(site_array)
    boot_f1s = []
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    for _ in range(CONFIG["bootstrap_draws"]):
        sampled_sites = rng.choice(unique_sites, size=len(unique_sites), replace=True)
        sampled_indices = np.concatenate([np.where(site_array == s)[0] for s in sampled_sites])
        if len(sampled_indices) > 0:
            b_true = y_true[sampled_indices]
            b_pred = final_preds[sampled_indices]
            boot_f1s.append(f1_score(b_true, b_pred, average="macro", labels=range(n_classes), zero_division=0))
            
    ci_low, ci_high = np.percentile(boot_f1s, [2.5, 97.5]) if boot_f1s else (macro_f1, macro_f1)
    return macro_f1, acc, act_rate, f"[{ci_low:.3f}, {ci_high:.3f}]"

f1_vis, acc_vis, act_vis, ci_vis = evaluate_policy_detailed(np.zeros(len(y_true), dtype=bool), fusion_preds, vis_preds_base, y_true, site_ids)
f1_both, acc_both, act_both, ci_both = evaluate_policy_detailed(np.ones(len(y_true), dtype=bool), fusion_preds, vis_preds_base, y_true, site_ids)
f1_oracle, acc_oracle, act_oracle, ci_oracle = evaluate_policy_detailed(np.ones(len(y_true), dtype=bool), oracle_sensor_preds, vis_preds_base, y_true, site_ids)

# ------------------------------------------------------------------------------
# THRESHOLD SWEEP & SEED-AVERAGED RANDOM DEFERRAL
# ------------------------------------------------------------------------------
sweep_results = []
for tau in CONFIG["tau_thresholds"]:
    defer = vis_conf < tau
    f1, acc, act, ci = evaluate_policy_detailed(defer, fusion_preds, vis_preds_base, y_true, site_ids)
    sweep_results.append({"tau": tau, "f1": f1, "acc": acc, "act": act, "ci": ci})

rand_results = []
for res in sweep_results:
    target_act = res["act"]
    seed_f1s = []
    for seed in CONFIG["random_seeds"]:
        rng_seed = np.random.default_rng(seed)
        rand_defer = rng_seed.random(len(y_true)) < target_act
        f1_r, _, _, _ = evaluate_policy_detailed(rand_defer, fusion_preds, vis_preds_base, y_true, site_ids)
        seed_f1s.append(f1_r)
    rand_results.append({"act": target_act, "f1": np.mean(seed_f1s)})

# ------------------------------------------------------------------------------
# PLOTTING THE PARETO CURVE
# ------------------------------------------------------------------------------
print("\n2. Generating Pareto Curve Plot...")
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

tau_acts = [r["act"] for r in sweep_results]
tau_f1s = [r["f1"] for r in sweep_results]
rand_acts = [r["act"] for r in rand_results]
rand_f1s = [r["f1"] for r in rand_results]

ax.plot(tau_acts, tau_f1s, color="#005B94", lw=2.5, label=r"Confidence Selector ($	au$ Sweep)")
ax.plot(rand_acts, rand_f1s, color="gray", linestyle=":", lw=2, label="Random Deferral (Seeds 0–4)")

ax.scatter([0.0], [f1_vis], color="#D9534F", s=120, zorder=5, label=f"Always-Visual ({f1_vis:.3f})")
ax.scatter([1.0], [f1_both], color="#2E7D32", s=120, zorder=5, label=f"Always-Both ({f1_both:.3f})")
ax.scatter([1.0], [f1_oracle], color="#8E44AD", s=140, marker="*", zorder=5, label=f"Oracle Sensor Choice ({f1_oracle:.3f})")

ax.set_title("SA-FARI Selection Frontier: Macro-F1 vs Audio Activation Rate", fontsize=11, fontweight="bold")
ax.set_xlabel("Audio Activation Rate (Inference Call Cost)", fontsize=10)
ax.set_ylabel("Macro-F1 Score", fontsize=10)
ax.set_xlim(-0.03, 1.05)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=8.5)

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/selector_pareto_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------
# TARGETED LOG SCRUBBING (Only exact experiment names for this split)
# ------------------------------------------------------------------------------
log_path = "results_log.csv"
current_experiment_names = [f"baseline_always_visual_{split_label}", f"baseline_always_both_{split_label}", f"oracle_sensor_choice_{split_label}"]
for res in sweep_results:
    t = res["tau"]
    current_experiment_names.append(f"conf_threshold_{t:.2f}_{split_label}")

new_rows = [
    {
        "experiment": f"baseline_always_visual_{split_label}", "question": CONFIG["question"], "metric": "macro_f1", 
        "value": round(f1_vis, 4), "interval": ci_vis, "number of events": n_clips, 
        "shift level": shift_level, "k": CONFIG["k"], "tag": CONFIG["tag"], "accuracy": round(acc_vis, 4), 
        "audio_used_rate": 0.0, "compute_proxy": 0.0
    },
    {
        "experiment": f"baseline_always_both_{split_label}", "question": CONFIG["question"], "metric": "macro_f1", 
        "value": round(f1_both, 4), "interval": ci_both, "number of events": n_clips, 
        "shift level": shift_level, "k": CONFIG["k"], "tag": CONFIG["tag"], "accuracy": round(acc_both, 4), 
        "audio_used_rate": 1.0, "compute_proxy": 1.0
    },
    {
        "experiment": f"oracle_sensor_choice_{split_label}", "question": CONFIG["question"], "metric": "macro_f1", 
        "value": round(f1_oracle, 4), "interval": ci_oracle, "number of events": n_clips, 
        "shift level": shift_level, "k": CONFIG["k"], "tag": CONFIG["tag"], "accuracy": round(acc_oracle, 4), 
        "audio_used_rate": 1.0, "compute_proxy": 1.0
    }
]

for res in sweep_results:
    t = res["tau"]
    new_rows.append({
        "experiment": f"conf_threshold_{t:.2f}_{split_label}", "question": CONFIG["question"], "metric": "macro_f1", 
        "value": round(res["f1"], 4), "interval": res["ci"], "number of events": n_clips, 
        "shift level": shift_level, "k": CONFIG["k"], "tag": CONFIG["tag"], "accuracy": round(res["acc"], 4),
        "audio_used_rate": round(res["act"], 4), "compute_proxy": round(res["act"], 4)
    })

new_df = pd.DataFrame(new_rows)

if os.path.exists(log_path):
    old_df = pd.read_csv(log_path)
    if "experiment" in old_df.columns:
        old_df = old_df[~old_df["experiment"].isin(current_experiment_names)]
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    combined_df = new_df

combined_df.to_csv(log_path, index=False)
print(f"Successfully scrubbed old logs and updated {log_path} with {n_clips} target clips for split {split_label}!")
