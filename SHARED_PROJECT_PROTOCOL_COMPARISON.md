# Shared project protocol comparison

Both tracks ask whether selectively acquiring an additional acoustic modality is worthwhile, but they make different operational decisions and must not share scores or empirical claims.

| Dimension | SA-FARI — Member 2 | Taiwan — Member 1 |
|---|---|---|
| Decision unit | Camera-triggered clip | Station-hour |
| Acquisition decision | Process audio or not | Activate recorder or not |
| Always-available modality | Camera | Camera |
| Additional modality | Audio | Acoustic recorder |
| Cost proxy | Percentage of clips processed | Percentage of recording time / recorder-on hours |
| Primary benefit metric | Incremental log-probability benefit / classification performance | Detection capture rate |
| Main reference policy | Camera uncertainty | Fixed source-derived schedule |
| Adaptive policy | Frozen source-selected selector | Target-updating adaptive schedule |
| Target feedback | None | Selected recorder outcomes update later decisions |
| Oracle | Retrospective clip-level oracle | Retrospective station-level oracle |
| Bootstrap unit | Site | Station |

SA-FARI evaluates a frozen pre-audio, clip-level selector and uses target labels only for evaluation. Taiwan's adaptive schedule retains online updates from selected target recorder outcomes; its threshold is source selected, but its later target actions are not feedback-free. Incremental log-probability benefit is not detection capture rate, and clip processing fraction is not recorder-on time. Consequently, these results can be compared as protocols, never combined as a single score or treated as direct empirical validation of one another.
