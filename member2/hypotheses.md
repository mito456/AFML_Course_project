# Research Hypotheses: Adaptive Sensor Selection for Wildlife Monitoring under Domain Shift

**Author / Responsibility:** Member 2  
**Project:** Adaptive Sensor Selection for Wildlife Monitoring under Domain Shift  
**Benchmark & Dataset:** SA-FARI Multi-Modal Wildlife Monitoring Benchmark  
**Core Reference Files:**  
- `safari/outputs/results/results_summary.csv`
- `safari/outputs/results/results_slices.csv`
- `safari/outputs/results/shift_results.csv`
- `safari/outputs/results/shift_drop_diff.csv`
- `safari/outputs/results/shortcut_results.csv`
- `safari/outputs/results/shortcut_night_results.csv`
- `safari/outputs/results/shortcut_placement_results.csv`
- `safari/outputs/results/results_per_species.csv`
- `safari/outputs/clip_events.csv`
- `safari/outputs/split.csv`
- `safari/outputs/test_meta.csv`

---

## 1. Q1 — Existing Expert Performance (M3 Baseline Hypotheses)

The following hypotheses from Milestone 3 (M3) evaluate the baseline unimodal expert models (`global_visual` / camera expert and `audio_only` / audio expert) under environmental distribution shifts and shortcut baselines. They are reproduced **verbatim** from M3:

### Hypothesis 1 (H1)
> **H1:** A visual classifier trained on camera trap footage will achieve substantially higher overall species recognition accuracy than an audio-only classifier on camera-triggered recordings, as camera traps are visually triggered and animals are frequently silent or call intermittently.

### Hypothesis 2a (H2a)
> **H2a:** When evaluated under spatial domain shift to unseen camera locations within the same geographic project ($L0 \to L1$), visual classifiers will experience a larger drop in classification performance than audio-only classifiers, due to visual background overfitting and site-specific background shortcuts.

### Hypothesis 2b (H2b)
> **H2b:** When evaluated under environmental domain shift to entirely new projects and ecosystems ($L1 \to L2$), audio-only classifiers will experience greater or comparable degradation relative to visual classifiers, as background biophony, acoustic soundscapes, and ambient environmental noise vary fundamentally across distinct regional ecosystems.

### Hypothesis 3 (H3)
> **H3:** Audio-only classifiers rely partially on non-acoustic environmental shortcuts (such as site-specific ambient acoustic signatures and diurnal activity patterns correlated with day/night conditions), but retain genuine acoustic predictive signal that significantly exceeds trivial metadata priors (such as project priors, joint project-night priors, and majority class baselines).

---

## 2. Q2 — Learning When Audio Is Useful

In the SA-FARI deployment setting, raw recordings are initiated by motion/passive infrared triggers at the camera trap, generating visual footage and synchronous audio. Processing audio streams continuously incurs unnecessary computation and energy costs, and can introduce acoustic noise when the target animal is silent. 

Under Question 2 (Q2), we investigate whether an adaptive selector policy can predict *prior to running the audio pipeline* whether invoking audio processing will provide complementary predictive value beyond the camera expert alone.

### Hypothesis 1 (H1)
> **H1:** A model trained on source-location data can learn to identify clips where audio provides additional predictive value beyond the camera, and retain some ability to do so at unseen locations.

*Note on Domain Shift:* Geographic and environmental domain shift between source training deployments and target deployment locations alters both visual background distributions and acoustic ambient properties. We explicitly acknowledge that geographic domain shift may reduce selector reliability at unseen target locations.

### Hypothesis 2 (H2)
> **H2:** Small amounts of labelled target-location data ($k=1$ and $k=3$ shots per class across seeds 0–4) may improve audio-usefulness estimation compared with the zero-label ($k=0$) setting.

---

## 3. Q3 — Adaptation and Selective Audio

Under Question 3 (Q3), we examine few-shot target adaptation of the unimodal experts and the interaction between visual adaptation and selective audio invocation.

### Hypothesis 3 (H3)
> **H3:** Adapting the camera expert using limited target labels ($k=1, 3$ per class, evaluated over random seeds 0–4) can improve camera predictions at new locations.

### Hypothesis 4 (H4)
> **H4:** Selectively processing audio can improve performance over the camera-only approach when audio contains complementary information.

### Hypothesis 5 (H5)
> **H5:** Combining camera adaptation with selective audio may provide complementary gains, or their benefits may overlap. This must be experimentally tested, not assumed.

---

## 4. Conditions Affecting Audio Usefulness (Hypotheses)

The following statements represent structured research hypotheses regarding environmental, observational, and taxonomic conditions that may govern the marginal utility of audio over camera predictions. In accordance with rigorous scientific evaluation, these are formulated as **hypotheses to be tested**, not established findings.

### Condition Hypothesis 1 (C-H1): Visual Ambiguity and Occlusion
> Under visually ambiguous conditions—such as when an animal is briefly visible, partially occluded, distant, or exiting the camera frame—the marginal benefit of invoking the audio expert over the camera expert alone will be higher than in unobstructed, prolonged visual encounters.

### Condition Hypothesis 2 (C-H2): Low-Light and Infrared (IR) Illumination
> Under degraded visual conditions indicated by the low-light/IR proxy `is_ir`, visual expert confidence and discrimination drop; consequently, audio processing will provide greater relative error correction and complementary signal than under daytime colour capture.

### Condition Hypothesis 3 (C-H3): Camera Placement and Micro-Habitat Geometry
> Camera placement geometry (e.g., arboreal canopy mounting vs. terrestrial ground mounting) affects visual and acoustic capture dynamics differently. In arboreal placements, dense foliage occludes visual fields while acoustic propagation remains relatively unobstructed, hypothesised to yield a higher rate of audio usefulness compared with open terrestrial placements.

### Condition Hypothesis 4 (C-H4): Species Vocality, Body Size, and Diurnal Activity Patterns
> The utility of selective audio processing will vary systematically across taxonomic orders and functional guilds. Specifically:
> 1. Highly vocal species and social species (e.g., certain primates and flocking birds) will exhibit higher audio complementarity than solitary, quiet species (e.g., felids or small rodents).
> 2. Small-bodied animals (whose low visual pixel area impedes camera classification) will benefit more from complementary acoustic cues when vocalisations or movement rustling sounds are present.

---

## 5. Architectural & Methodological Constraints

To ensure rigorous, leak-free experimental design that strictly adheres to real-world edge hardware deployment constraints, the following rules govern selector design and slice evaluation:

### 5.1 Camera-First Policy Framing
SA-FARI sensors operate with camera-triggered recording. The operational policy is strictly **camera-first**: the camera expert runs on incoming visual frames, and the adaptive selector decides whether to invoke audio processing and combine multimodal predictions or remain unimodal (camera-only).

### 5.2 Admissible Selector Inputs vs. Post-Hoc Analysis Variables

| Variable | Type / Origin | Selector Input (Admissible)? | Post-Hoc Analysis Only? | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Visual features / embeddings** (`visual_emb`) | Pre-audio visual model | **Yes** | No | Extracted from camera frames before audio processing. |
| **Camera classifier confidence / entropy** (`conf_camera`) | Camera expert prediction | **Yes** | No | Fully available from initial camera inference. |
| **Low-light / IR proxy** (`is_ir`) | Hardware sensor metadata | **Yes** | No | Hardware flag indicating night IR illumination mode. |
| **Camera placement** (`placement`) | Site setup configuration | **Yes** | No | Known at deployment time (e.g., arboreal vs. terrestrial). |
| **Audio features / embeddings** (`audio_emb`) | Audio pipeline | **NO** | Post-Hoc | Requires processing audio; defeats the purpose of selective audio invocation. |
| **Audio-derived scores / probs** (`audioset_probs`) | Audio pipeline | **NO** | Post-Hoc | Cannot be used before audio inference occurs. |
| **Animal masklet area** (`ana_mean_area`) | Ground-truth annotation | **NO** | **Yes** | Oracle annotation; unavailable at real-world test time. |
| **Visibility fraction** (`ana_visible_frac`) | Ground-truth annotation | **NO** | **Yes** | Oracle annotation; unavailable at real-world test time. |
| **Frame exit indicator** (`ana_exits_frame`) | Ground-truth annotation | **NO** | **Yes** | Oracle annotation; unavailable at real-world test time. |
| **Target sound presence** (`ana_target_sound`) | Ground-truth annotation | **NO** | **Yes** | Oracle acoustic annotation. |
| **True species label** (`species`) | Ground-truth annotation | **NO** | **Yes** | Evaluation target label only. |
| **Site ID / Location ID** (`location_id`) | Deployment metadata | **NO** | Post-Hoc | Unseen at target locations; using site ID creates site-shortcut overfitting. |

### 5.3 Technical Distinctions & Experimental Standards
- **`is_ir` Flag:** `is_ir` is strictly an indicator of infrared/low-light illumination mode at the sensor hardware level, **not a definitive night label** (e.g., deep twilight, heavy forest canopy shade, or sensor switching can trigger IR).
- **Adaptation Protocol:** Target few-shot adaptation evaluations must strictly adhere to $k \in \{0, 1, 3\}$ shots per class across five reproducible seeds (`seeds = [0, 1, 2, 3, 4]`).
- **Separation of Hypotheses from Findings:** Hypotheses must remain clearly distinguished from empirical results throughout experimental analysis.
