# Camera expert: logistic regression on BioCLIP embeddings of frames at 1 fps
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUT

C = 0.01


# Per-clip list of 1-fps frame embeddings (4-frame mean if a clip has no frames)
def load_frames(idx):
    z = np.load(OUT / "visual_frames_1fps.npz", allow_pickle=True)
    assert (z["video_name"] == idx.video_name.to_numpy()).all(), "visual_frames_1fps.npz order differs from feat_index.csv"
    V4 = np.load(OUT / "visual_emb.npy")
    emb, off = z["emb"], z["offset"]
    return [emb[off[i]:off[i + 1]] if off[i + 1] > off[i] else V4[i:i + 1] for i in range(len(idx))]


# Each clip gets equal total weight over its frames; the clip's prediction is its most confident frame
def fit_proba(frames, y_tr, tr, te, classes, c=C, w_tr=None):
    w_tr = np.ones(len(tr)) if w_tr is None else np.asarray(w_tr, float)
    Xf = np.concatenate([frames[i] for i in tr])
    yf = np.concatenate([[lab] * len(frames[i]) for i, lab in zip(tr, y_tr)])
    wf = np.concatenate([[w / len(frames[i])] * len(frames[i]) for i, w in zip(tr, w_tr)])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=c))
    m.fit(Xf, yf, logisticregression__sample_weight=wf)
    cols = [list(classes).index(k) for k in m.classes_]
    out = np.zeros((len(te), len(classes)))
    for j, i in enumerate(te):
        p = m.predict_proba(frames[i])
        out[j, cols] = p[p.max(1).argmax()]
    return out


def fit_predict(frames, y_tr, tr, te, classes, c=C):
    return np.array(classes)[fit_proba(frames, y_tr, tr, te, classes, c).argmax(1)]
