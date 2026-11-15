# Shared protocol comparison

| Protocol dimension | Member 1: Station-level | Member 2: Clip-level |
|---|---|---|
| Decision unit | Station-hour | Camera-triggered clip |
| Acquisition decision | Whether recorder is active | Whether audio is processed/used |
| Always-available modality | Camera | Camera |
| Additional modality | Acoustic recorder | Audio |
| Primary evaluation metric | Detection capture rate | Incremental log loss / classification performance |
| Acquisition cost proxy | Recording budget (%% recorder-on) | Audio processing rate (%% clips processed) |
| Random baseline | Analytical expected capture or implemented random schedule (report-specific) | Empirical random selection |
| Source-only baseline | Fixed source-derived schedule | Camera uncertainty / source-trained selector |
| Adaptive method | Target-updating adaptive schedule (selected-outcome feedback assumed) | Frozen source-trained selector |
| Oracle | Retrospective target-label oracle | Retrospective clip-level oracle |
| Target feedback | Selected recorder outcomes update future decisions | No target-label feedback |
| Bootstrap unit | Station | Site |

These are protocol alignments, not equivalent algorithms. Member 1 makes station-hour recording decisions; Member 2 makes clip-level audio-processing decisions. Their cost proxies and primary benefit metrics differ. Member 1's adaptive policy uses selected target recorder outcomes for online updating, whereas Member 2 target evaluation remains frozen. Member 2 covers only camera-triggered clips and cannot estimate audio value for events never detected by the camera.
