"""E020 1단계 검정: 신고자 음성에 전사 텍스트가 이미 담은 것 이상의 증상 신호가 있는가?

    python runs/E020_audio/stage1.py            # 결과: stage1.txt, stage1.json, stage1_oof.npz

통화 전체를 들은 시점(마지막 순간) 하나만 본다. 통화 단위 5겹 교차검증, fit 통화 6,000건(sample_ids.txt)만 쓴다.
dev / Validation 은 읽지 않는다.

설계 수정 (2026-11-13, 본 실행 결과를 보기 전):
  검토 담당이 같은 6,000통화·같은 겹에서 원 설계의 LightGBM T 기준선이 9종 평균 AUC 0.8701 로
  T_raw(텍스트 확률 그대로) 0.8848 보다 0.015 낮다는 것을 측정했다. 그래서 1차 분류기를 로지스틱 회귀로
  바꾸고, T_raw 대비 차이도 판정 조건에 넣었다. LightGBM 은 2차(잔차 부스팅)와 기록용(원 설계)으로만 남긴다.

특징
  T  : E011 교차 학습 확률표(runs/E011_oof/fold{k}/oof_prefix.npz)에서 통화의 마지막 순간 9종 확률 -> 로짓.
       이 표의 각 통화 확률은 그 통화를 학습에 쓰지 않은 모델의 출력이다.
  A  : 신고자 발화 임베딩(emb.npz) 중 한 층의 발화 평균·표준편차(ddof=0) 1,536차원
       -> 표준화 -> PCA 64 (둘 다 학습 겹의 "발화가 1개 이상인 통화"로만 fit)
       + 발화 0개 표시 특징 1개. 발화 0개 통화는 PCA 공간에서 0 벡터(= 학습 겹 평균)로 둔다.
       1차 = 8번째 층(hidden_states[8]). 4·12번째 층은 "탐색"으로만 따로 적는다.

분류기 (모두 같은 겹: KFold(5, shuffle=True, random_state=2026), 통화 단위)
  1차   LR    : 증상별 LogisticRegression(C=1, lbfgs, max_iter=2000), 입력은 학습 겹에서 fit 한 StandardScaler 로 표준화
              LR_T, LR_T+A, LR_T+A_shuf(대조군: 음성 블록만 통화 간 섞음, seed 2026), LR_A
  2차   RES   : LightGBM(고정 설정) + init_score = 자기 증상의 텍스트 로짓 (잔차 부스팅). RES_T, RES_T+A
  기록  ORIG  : 원 설계 LightGBM (init_score 없음). ORIG_T, ORIG_T+A
  참고  T_raw : 분류기 없이 E011 확률 그대로

지표: 증상별 AUC, 최대 F1(겹 밖 예측 전체에서 임계를 고름 — 비교 대상 모두 같은 낙관), 9종 평균.
      AUC 차이에 통화 단위 짝 부트스트랩 95% 구간(모든 모델에 같은 재표집 색인, 2,000회, seed 2026).

사전 등록 판정 (수정본, 1차 LR 기준. 2차·원 설계는 판정에 쓰지 않음): 다음 중 하나면 2단계로 간다.
  (a) 평균 AUC: T+A − T >= 0.005 이고 그 하한 > 0, 그리고 T+A − T_raw 하한 > 0
  (b) 어느 한 증상: T+A − T >= 0.02 이고 그 하한 > 0, 그 증상의 T+A − T_raw 하한 > 0,
      그리고 그 증상의 대조군 − T 구간이 0 을 포함
해석용: 텍스트가 512토큰에서 잘린 통화를 뺀 1차 T+A − T 평균 AUC 차이도 낸다.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # runs/E020_audio
ROOT = HERE.parent.parent                         # mission4_symptom
sys.path.insert(0, str(ROOT))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4.data import label_matrix, load_cache, prefix_texts, read_ids  # noqa: E402
from m4.simulate import PrefixTable  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

SEED = 2026
N_FOLDS = 5
PCA_DIM = 64
PRIMARY_LAYER = 8
EXPLORE_LAYERS = (4, 12)
N_BOOT = 2000
TEXT_MAX_TOKENS = 512
LGB_PARAMS = dict(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=40,
                  colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)
LR_PARAMS = dict(C=1.0, max_iter=2000, solver="lbfgs")
# 사전 등록 판정 기준
GATE_MEAN_DIFF = 0.005
GATE_CLASS_DIFF = 0.02

# 부트스트랩에 넣는 모델과 구간을 낼 차이 (앞 - 뒤)
BOOT_MODELS = ["T_raw", "LR_T", "LR_T+A", "LR_T+A_shuf", "RES_T", "RES_T+A", "ORIG_T", "ORIG_T+A"]
BOOT_PAIRS = [("LR_T+A", "LR_T"),            # 1차 판정
              ("LR_T+A", "T_raw"),           # 1차 판정
              ("LR_T+A_shuf", "LR_T"),       # 대조군 (0 이어야 정상)
              ("RES_T+A", "T_raw"),          # 2차
              ("RES_T+A", "RES_T"),          # 2차 참고
              ("ORIG_T+A", "ORIG_T"),        # 원 설계 기록
              ("LR_T", "T_raw")]             # 1차 기준선이 텍스트 확률 그대로보다 얼마나 잃는지
LABEL = {"T_raw": "T_raw", "LR_T": "T", "LR_T+A": "T+A", "LR_T+A_shuf": "대조군", "LR_A": "A",
         "RES_T": "잔차T", "RES_T+A": "잔차T+A", "ORIG_T": "원T", "ORIG_T+A": "원T+A",
         "LR_T+A_L4": "T+A_L4", "LR_T+A_L12": "T+A_L12"}


# ----------------------------------------------------------------------------- 데이터
def load_text_features(ids: list) -> tuple:
    """E011 교차 확률표에서 통화별 마지막 순간 확률 (n, 9), 표의 라벨 (n, 9), 표의 순간 수 (n,), 등장 횟수 (n,)."""
    pos = {c: i for i, c in enumerate(ids)}
    probs = np.full((len(ids), len(SYMPTOM_CLASSES)), np.nan, dtype=np.float64)
    labels = np.full((len(ids), len(SYMPTOM_CLASSES)), -1, dtype=np.int64)
    n_steps = np.zeros(len(ids), dtype=np.int64)
    hits = np.zeros(len(ids), dtype=np.int64)
    for k in range(N_FOLDS):
        tab = PrefixTable.load(ROOT / f"runs/E011_oof/fold{k}/oof_prefix.npz")
        for j, cid in enumerate(tab.call_ids):
            i = pos.get(cid)
            if i is None:
                continue
            p = tab.call(j)
            probs[i] = p[-1]                     # 마지막 순간 = 통화 전체를 들은 시점
            labels[i] = tab.labels[j]
            n_steps[i] = len(p)
            hits[i] += 1
    return probs, labels, n_steps, hits


def text_truncation(calls: list) -> tuple:
    """E011 과 같은 토크나이저·입력 형식으로 통화 전체 텍스트의 토큰 수를 센다.

    잘림 = 특수 토큰 포함 토큰 수 > 512 (E011 은 오른쪽을 잘라 앞부분만 봤다).
    E011 메타의 truncated_moments 기준(잘린 뒤 길이 >= 512, 정확히 512 인 것도 셈)도 같이 돌려준다.
    """
    from transformers import AutoTokenizer
    cfgs = [json.loads((ROOT / f"runs/E011_oof/fold{k}/run_config.json").read_text(encoding="utf-8"))
            for k in range(N_FOLDS)]
    backbones = {c["backbone"] for c in cfgs}
    fmts = {c["fmt"] for c in cfgs}
    assert len(backbones) == 1 and len(fmts) == 1, (backbones, fmts)
    assert {c["max_length"] for c in cfgs} == {TEXT_MAX_TOKENS}
    tok = AutoTokenizer.from_pretrained(str(ROOT / backbones.pop()), local_files_only=True)
    fmt = fmts.pop()
    texts = [prefix_texts(c, fmt)[-1] for c in calls]
    n_tok = np.array([len(x) for x in tok(texts, truncation=False)["input_ids"]])
    return n_tok > TEXT_MAX_TOKENS, n_tok >= TEXT_MAX_TOKENS, n_tok


def load_audio_raw(ids: list, layer: int, emb_path: Path) -> tuple:
    """emb.npz 에서 통화별 [발화 평균, 발화 표준편차] (n, 2*768) 와 임베딩된 발화 수 (n,)."""
    z = np.load(emb_path, allow_pickle=False)
    layers = [int(x) for x in z["layers"]]
    li = layers.index(layer)
    emb = z["emb"][:, li, :].astype(np.float64)          # (N, 768)
    call_ids = [str(x) for x in z["call_ids"]]
    row_call = z["call"]
    pos = {c: i for i, c in enumerate(ids)}
    owner = np.array([pos[call_ids[c]] for c in row_call], dtype=np.int64)   # 발화 -> 표본 순서 통화
    n = len(ids)
    cnt = np.bincount(owner, minlength=n)
    s1 = np.zeros((n, emb.shape[1]))
    s2 = np.zeros((n, emb.shape[1]))
    np.add.at(s1, owner, emb)
    np.add.at(s2, owner, emb ** 2)
    has = cnt > 0
    mean = np.zeros_like(s1)
    mean[has] = s1[has] / cnt[has, None]
    var = np.zeros_like(s1)
    var[has] = np.maximum(s2[has] / cnt[has, None] - mean[has] ** 2, 0.0)
    raw = np.hstack([mean, np.sqrt(var)])                 # 발화 0개 통화는 0 (어차피 아래에서 0 벡터로 덮음)
    in_npz = set(call_ids)
    return raw, cnt, has, sum(1 for c in ids if c not in in_npz)


def audio_block(raw: np.ndarray, has: np.ndarray, tr: np.ndarray) -> tuple:
    """표준화·PCA 를 학습 겹(tr)의 발화 있는 통화로만 fit 하고 전체 통화에 적용한다. -> (n, 64+1), 설명 분산 합."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    fit_rows = tr[has[tr]]
    sc = StandardScaler().fit(raw[fit_rows])
    # 본 실행에서는 항상 64. 통화 수가 아주 적은 코드 점검(20통화 임베딩)에서만 줄어든다.
    n_comp = min(PCA_DIM, len(fit_rows) - 1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(sc.transform(raw[fit_rows]))
    Z = pca.transform(sc.transform(raw))
    Z[~has] = 0.0                                         # 발화 0개: 0 벡터 + 표시 특징
    X = np.hstack([Z, (~has).astype(np.float64)[:, None]])
    return X, float(pca.explained_variance_ratio_.sum())


# ----------------------------------------------------------------------------- 분류기
def fit_predict(kind: str, X: np.ndarray, Y: np.ndarray, T: np.ndarray, tr: np.ndarray,
                te: np.ndarray, warn_count: list) -> np.ndarray:
    """한 겹에서 증상별 이진 분류기 9개를 학습하고 te 확률 (len(te), 9) 를 낸다.

    lr        : 학습 겹 StandardScaler → LogisticRegression (1차)
    lgb       : LightGBM 그대로 (원 설계)
    lgb_resid : LightGBM, init_score = 자기 증상 텍스트 로짓 T[:, j]. 예측 = 원점수 + T[te, j] → 시그모이드.
                (LightGBM predict 는 init_score 를 더하지 않으므로 직접 더한다.)
    """
    from scipy.special import expit
    out = np.zeros((len(te), Y.shape[1]))
    if kind == "lr":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for j in range(Y.shape[1]):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = LogisticRegression(**LR_PARAMS).fit(Xtr, Y[tr, j])
            warn_count[0] += sum(issubclass(x.category, ConvergenceWarning) for x in w)
            out[:, j] = clf.predict_proba(Xte)[:, 1]
        return out
    from lightgbm import LGBMClassifier
    for j in range(Y.shape[1]):
        clf = LGBMClassifier(**LGB_PARAMS)
        if kind == "lgb":
            clf.fit(X[tr], Y[tr, j])
            out[:, j] = clf.predict_proba(X[te])[:, 1]
        elif kind == "lgb_resid":
            clf.fit(X[tr], Y[tr, j], init_score=T[tr, j])
            out[:, j] = expit(clf.predict(X[te], raw_score=True) + T[te, j])
        else:
            raise ValueError(kind)
    return out


def run_cv(kind: str, T: np.ndarray, A_folds, Y: np.ndarray, folds: list, use_text: bool,
           warn_count: list) -> np.ndarray:
    """A_folds: 겹마다 미리 만든 음성 블록 목록 (None 이면 음성 없음)."""
    oof = np.full(Y.shape, np.nan)
    for f, (tr, te) in enumerate(folds):
        parts = ([T] if use_text else []) + ([A_folds[f]] if A_folds is not None else [])
        X = np.hstack(parts)
        oof[te] = fit_predict(kind, X, Y, T, tr, te, warn_count)
    assert not np.isnan(oof).any()
    return oof


# ----------------------------------------------------------------------------- 지표
def auc_matrix(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """열마다 순위합(Mann-Whitney) AUC. 동점은 평균 순위 — sklearn roc_auc_score 와 같은 값. -> (9,)"""
    from scipy.stats import rankdata
    R = rankdata(P, axis=0)
    pos = Y == 1
    n_pos = pos.sum(0)
    n_neg = len(Y) - n_pos
    return ((R * pos).sum(0) - n_pos * (n_pos + 1) / 2) / np.maximum(n_pos * n_neg, 1)


def max_f1(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, score)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.max())


def per_class(pred: np.ndarray, Y: np.ndarray) -> dict:
    auc = [float(x) for x in auc_matrix(pred, Y)]
    f1 = [max_f1(pred[:, j], Y[:, j]) for j in range(Y.shape[1])]
    return {"auc": auc, "f1": f1, "auc_mean": float(np.mean(auc)), "f1_mean": float(np.mean(f1))}


def paired_bootstrap(preds: dict, Y: np.ndarray, n_boot: int) -> dict:
    """통화 단위 짝 부트스트랩. BOOT_MODELS 모두에 같은 재표집 색인을 쓴다.

    BOOT_PAIRS 의 AUC 차이(앞 − 뒤)마다 증상별 95% 구간과, 회차마다 9종 차이를 평균한 값의 95% 구간을 낸다.
    """
    rng = np.random.default_rng(SEED)
    n, k = Y.shape
    boot = {m: np.zeros((n_boot, k)) for m in BOOT_MODELS}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = Y[idx]
        for m in BOOT_MODELS:
            boot[m][b] = auc_matrix(preds[m][idx], yb)
    out = {}
    for a, c in BOOT_PAIRS:
        d = boot[a] - boot[c]                           # (n_boot, 9)
        dm = d.mean(axis=1)
        out[f"{a} - {c}"] = {
            "mean_ci": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            "class_ci": [[float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]
                         for j in range(k)]}
    return out


# ----------------------------------------------------------------------------- 메인
def main() -> int:
    import argparse
    from sklearn.model_selection import KFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(HERE / "emb.npz"))
    ap.add_argument("--out-prefix", default="stage1", help="출력 파일 이름 앞부분 (HERE 아래)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="코드 점검용으로만 줄인다")
    ap.add_argument("--smoke", action="store_true", help="코드 점검(20통화 임베딩): 음성 관련 assert 를 끈다")
    args = ap.parse_args()
    emb_path = Path(args.emb)
    t0 = time.time()

    ids = read_ids(HERE / "sample_ids.txt")
    assert len(ids) == len(set(ids)) == 6000
    fit_ids = set(read_ids(ROOT / "runs/cache/fit_ids.txt"))
    assert set(ids) <= fit_ids, "표본에 fit 이 아닌 통화가 있다"

    # 라벨: 캐시(표본 순서로 다시 정렬)
    calls = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", ids)}
    assert len(calls) == len(ids)
    calls = [calls[c] for c in ids]
    Y = label_matrix(calls).astype(np.int64)

    # T
    P, tab_labels, n_steps, hits = load_text_features(ids)
    checks = {
        "text_calls_found": int((hits == 1).sum()), "text_calls_missing": int((hits == 0).sum()),
        "text_calls_dup": int((hits > 1).sum()),
        "label_mismatch_vs_E011": int((tab_labels != Y).any(axis=1).sum()),
        "step_count_mismatch_vs_cache": int(sum(n_steps[i] != calls[i].n_utt for i in range(len(ids)))),
    }
    assert checks["text_calls_found"] == len(ids), checks
    assert checks["label_mismatch_vs_E011"] == 0, checks
    assert checks["step_count_mismatch_vs_cache"] == 0, checks
    eps = 1e-6
    Pc = np.clip(P, eps, 1 - eps)
    T = np.log(Pc / (1 - Pc))

    # 텍스트 잘림 (해석용)
    trunc, trunc_meta_rule, n_tok = text_truncation(calls)
    checks.update({"text_truncated_calls": int(trunc.sum()),
                   "text_truncated_calls_meta_rule_ge512": int(trunc_meta_rule.sum()),
                   "text_tokens_median": float(np.median(n_tok))})

    # A (1차 층)
    emb_meta = json.loads(Path(str(emb_path) + ".meta.json").read_text(encoding="utf-8"))
    raw, cnt, has, not_in_npz = load_audio_raw(ids, PRIMARY_LAYER, emb_path)
    checks.update({"audio_missing_wav_calls": int(emb_meta["missing_wav_calls"]),
                   "audio_calls_not_in_npz": not_in_npz, "audio_calls_zero_utts": int((~has).sum()),
                   "audio_utts_total": int(cnt.sum()), "audio_utts_per_call_mean": float(cnt.mean()),
                   "audio_utts_per_call_median": float(np.median(cnt))})
    if not args.smoke:
        assert emb_meta["missing_wav_calls"] == 0, "wav 를 못 찾은 통화가 있다"
        assert not_in_npz == 0, "emb.npz 에 없는 표본 통화가 있다"
    print("점검:", json.dumps(checks, ensure_ascii=False), flush=True)

    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(ids))))

    # 겹마다 음성 블록을 한 번만 만든다 (학습 겹으로만 fit). 대조군은 음성 원특징 행(표시 포함)을 통화끼리 섞는다.
    perm = np.random.default_rng(SEED).permutation(len(ids))
    blocks, evr = {}, {}
    sources = {"L8": (raw, has), "L8_shuf": (raw[perm], has[perm])}
    for layer in EXPLORE_LAYERS:
        r_l, _, h_l, _ = load_audio_raw(ids, layer, emb_path)
        sources[f"L{layer}"] = (r_l, h_l)
    for key, (r, h) in sources.items():
        res = [audio_block(r, h, tr) for tr, _ in folds]
        blocks[key] = [x for x, _ in res]
        evr[key] = float(np.mean([e for _, e in res]))
    print(f"  음성 블록 완료 ({time.time() - t0:.0f}s)", flush=True)

    warn_count = [0]
    preds = {"T_raw": P}
    plan = [("LR_T", "lr", True, None), ("LR_T+A", "lr", True, "L8"), ("LR_T+A_shuf", "lr", True, "L8_shuf"),
            ("LR_A", "lr", False, "L8"),
            ("RES_T", "lgb_resid", True, None), ("RES_T+A", "lgb_resid", True, "L8"),
            ("ORIG_T", "lgb", True, None), ("ORIG_T+A", "lgb", True, "L8"),
            # 탐색 (판정에 쓰지 않음): 1차 분류기 그대로 층만 바꿈
            ("LR_T+A_L4", "lr", True, "L4"), ("LR_T+A_L12", "lr", True, "L12")]
    for name, kind, use_text, src in plan:
        preds[name] = run_cv(kind, T, blocks[src] if src else None, Y, folds, use_text, warn_count)
        print(f"  {name} 완료 ({time.time() - t0:.0f}s)", flush=True)

    metrics = {m: per_class(p, Y) for m, p in preds.items()}
    boot = paired_bootstrap(preds, Y, args.n_boot)
    print(f"  부트스트랩 완료 ({time.time() - t0:.0f}s)", flush=True)

    def point(a: str, c: str, rows=None) -> dict:
        """전체(또는 rows) 겹 밖 예측에서 AUC 차이 점추정."""
        if rows is None:
            da = np.asarray(metrics[a]["auc"]) - np.asarray(metrics[c]["auc"])
        else:
            da = auc_matrix(preds[a][rows], Y[rows]) - auc_matrix(preds[c][rows], Y[rows])
        return {"auc_mean_diff": float(da.mean()), "auc_class_diff": [float(x) for x in da]}

    diffs = {f"{a} - {c}": point(a, c) for a, c in BOOT_PAIRS}
    keep = ~trunc
    trunc_analysis = {
        "n_truncated": int(trunc.sum()), "n_kept": int(keep.sum()),
        "LR_T+A - LR_T_all": diffs["LR_T+A - LR_T"]["auc_mean_diff"],
        "LR_T+A - LR_T_without_truncated": point("LR_T+A", "LR_T", keep)["auc_mean_diff"],
        # 조정자가 말한 164건은 E011 메타 기준(>=512, 정확히 512토큰 3건 포함). 그 기준으로 뺀 값도 같이 적는다.
        "n_truncated_ge512": int(trunc_meta_rule.sum()),
        "LR_T+A - LR_T_without_ge512": point("LR_T+A", "LR_T", ~trunc_meta_rule)["auc_mean_diff"],
        # 잘린 통화만의 차이는 164건 안팎이라 흔들림이 크다 (참고)
        "LR_T+A - LR_T_truncated_only": (point("LR_T+A", "LR_T", trunc)["auc_mean_diff"]
                                         if trunc.sum() >= 20 and (Y[trunc].sum(0) > 0).all()
                                         and (Y[trunc].sum(0) < trunc.sum()).all() else None),
    }
    explore = {m: {"auc_mean": metrics[m]["auc_mean"], "f1_mean": metrics[m]["f1_mean"],
                   **point(m, "LR_T")} for m in ("LR_T+A_L4", "LR_T+A_L12")}

    # 사전 등록 판정 (수정본, 1차 LR 기준)
    d_ta_t, d_ta_raw, d_ctrl = diffs["LR_T+A - LR_T"], diffs["LR_T+A - T_raw"], diffs["LR_T+A_shuf - LR_T"]
    b_ta_t, b_ta_raw, b_ctrl = boot["LR_T+A - LR_T"], boot["LR_T+A - T_raw"], boot["LR_T+A_shuf - LR_T"]
    cond_a = bool(d_ta_t["auc_mean_diff"] >= GATE_MEAN_DIFF and b_ta_t["mean_ci"][0] > 0
                  and b_ta_raw["mean_ci"][0] > 0)
    cond_b = []
    for j, name in enumerate(SYMPTOM_CLASSES):
        lo_c, hi_c = b_ctrl["class_ci"][j]
        if (d_ta_t["auc_class_diff"][j] >= GATE_CLASS_DIFF and b_ta_t["class_ci"][j][0] > 0
                and b_ta_raw["class_ci"][j][0] > 0 and lo_c <= 0 <= hi_c):
            cond_b.append(name)
    ctrl_ci = b_ctrl["mean_ci"]
    decision = {
        "cond_a_mean": cond_a, "cond_b_classes": cond_b, "go_stage2": bool(cond_a or cond_b),
        "control_mean_ci_contains_0": bool(ctrl_ci[0] <= 0 <= ctrl_ci[1]),
        "control_class_ci_excluding_0": [SYMPTOM_CLASSES[j] for j, (lo, hi) in enumerate(b_ctrl["class_ci"])
                                         if lo > 0 or hi < 0]}

    result = {"design_note": "설계 수정: 결과 보기 전, 검토 담당의 T 기준선 측정(0.8701 vs T_raw 0.8848)에 따라 "
                             "1차 분류기를 로지스틱으로 변경",
              "n_calls": len(ids), "prevalence": [float(x) for x in Y.mean(0)], "checks": checks,
              "emb": str(emb_path), "n_boot": args.n_boot, "pca_explained_var_mean": evr,
              "lr_params": LR_PARAMS, "lr_convergence_warnings": warn_count[0], "lgb_params": LGB_PARAMS,
              "folds": "KFold(5, shuffle=True, random_state=2026), 통화 단위", "primary_layer": PRIMARY_LAYER,
              "metrics": metrics, "diffs": diffs, "bootstrap": boot, "truncation": trunc_analysis,
              "explore": explore, "decision": decision, "seconds": round(time.time() - t0, 1)}
    pre = args.out_prefix
    (HERE / f"{pre}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(HERE / f"{pre}_oof.npz", call_ids=np.asarray(ids), labels=Y, text_truncated=trunc,
                        n_audio_utts=cnt, **{m.replace("+", "p"): p for m, p in preds.items()})
    text = render(result)
    (HERE / f"{pre}.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ----------------------------------------------------------------------------- 출력
def render(r: dict) -> str:
    M, B, D = r["metrics"], r["bootstrap"], r["diffs"]
    L = []
    L.append(r["design_note"])
    L.append("")
    L.append("E020 1단계: 신고자 음성이 전사 텍스트 이상의 증상 신호를 담는가 (통화 전체 시점)")
    L.append(f"표본 fit 통화 {r['n_calls']}건, {r['folds']}, 1차 음성 층 = hidden_states[{r['primary_layer']}], "
             f"부트스트랩 {r['n_boot']}회")
    c = r["checks"]
    L.append(f"점검: T 표에 있는 통화 {c['text_calls_found']} (없음 {c['text_calls_missing']}, 중복 {c['text_calls_dup']}), "
             f"E011 라벨 불일치 {c['label_mismatch_vs_E011']}, 순간 수 불일치 {c['step_count_mismatch_vs_cache']}")
    L.append(f"      임베딩 발화 {c['audio_utts_total']}개 (통화당 평균 {c['audio_utts_per_call_mean']:.2f}, "
             f"중앙 {c['audio_utts_per_call_median']:.0f}), 발화 0개 통화 {c['audio_calls_zero_utts']}, "
             f"wav 없음 {c['audio_missing_wav_calls']}, emb.npz 에 없는 통화 {c['audio_calls_not_in_npz']}")
    L.append(f"      텍스트 512토큰 초과(잘림) {c['text_truncated_calls']}건 "
             f"(E011 메타 기준 >=512: {c['text_truncated_calls_meta_rule_ge512']}건)")
    ev = r["pca_explained_var_mean"]
    L.append("      PCA 64 설명 분산(겹 평균): " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items())
             + f"; 로지스틱 수렴 경고 {r['lr_convergence_warnings']}회")
    L.append("")

    def ci(key: str, j=None) -> str:
        lo, hi = B[key]["mean_ci"] if j is None else B[key]["class_ci"][j]
        return f"[{lo:+.4f}, {hi:+.4f}]"

    def dv(key: str, j=None) -> float:
        return D[key]["auc_mean_diff"] if j is None else D[key]["auc_class_diff"][j]

    def table(title: str, cols: list, pairs: list) -> None:
        L.append(title)
        L.append(f"{'증상':<6}{'양성률':>7}" + "".join(f"{LABEL[x]:>9}" for x in cols)
                 + "".join(f"  {LABEL[a]}−{LABEL[b]:<8}{'95% 구간':<19}" for a, b in pairs))
        for j, name in enumerate(SYMPTOM_CLASSES):
            L.append(f"{name:<6}{r['prevalence'][j]:>7.3f}" + "".join(f"{M[x]['auc'][j]:>9.4f}" for x in cols)
                     + "".join(f"  {dv(f'{a} - {b}', j):>+8.4f} {ci(f'{a} - {b}', j):<21}" for a, b in pairs))
        L.append(f"{'평균':<6}{'':>7}" + "".join(f"{M[x]['auc_mean']:>9.4f}" for x in cols)
                 + "".join(f"  {dv(f'{a} - {b}'):>+8.4f} {ci(f'{a} - {b}'):<21}" for a, b in pairs))
        L.append("")

    table("[1차 AUC] 로지스틱 회귀. T_raw = 분류기 없이 E011 확률 그대로",
          ["T_raw", "LR_T", "LR_T+A", "LR_T+A_shuf", "LR_A"],
          [("LR_T+A", "LR_T"), ("LR_T+A", "T_raw"), ("LR_T+A_shuf", "LR_T")])
    table("[2차 AUC] LightGBM 잔차 부스팅 (init_score = 자기 증상 텍스트 로짓). 판정에 쓰지 않음",
          ["T_raw", "RES_T", "RES_T+A"], [("RES_T+A", "T_raw"), ("RES_T+A", "RES_T")])
    table("[원 설계 AUC] LightGBM (init_score 없음). 기록용, 판정에 쓰지 않음",
          ["T_raw", "ORIG_T", "ORIG_T+A"], [("ORIG_T+A", "ORIG_T"), ("LR_T", "T_raw")])

    fcols = ["T_raw", "LR_T", "LR_T+A", "LR_T+A_shuf", "LR_A", "RES_T", "RES_T+A", "ORIG_T", "ORIG_T+A"]
    L.append("[최대 F1] 겹 밖 예측 전체에서 증상별 임계를 고른 값 (모든 열에 같은 낙관)")
    L.append(f"{'증상':<6}" + "".join(f"{LABEL[x]:>9}" for x in fcols))
    for j, name in enumerate(SYMPTOM_CLASSES):
        L.append(f"{name:<6}" + "".join(f"{M[x]['f1'][j]:>9.4f}" for x in fcols))
    L.append(f"{'평균':<6}" + "".join(f"{M[x]['f1_mean']:>9.4f}" for x in fcols))
    L.append("")

    t = r["truncation"]
    L.append("[해석용] 텍스트 잘림 통화 제외")
    L.append(f"  1차 T+A − T 평균 AUC 차이: 전체 {t['LR_T+A - LR_T_all']:+.4f}, "
             f"잘린 {t['n_truncated']}건(>512토큰) 제외({t['n_kept']}건) {t['LR_T+A - LR_T_without_truncated']:+.4f}, "
             f">=512 {t['n_truncated_ge512']}건 제외 {t['LR_T+A - LR_T_without_ge512']:+.4f}"
             + (f", 잘린 통화만 {t['LR_T+A - LR_T_truncated_only']:+.4f} (표본 작음, 참고)"
                if t["LR_T+A - LR_T_truncated_only"] is not None else ""))
    L.append("")
    L.append("[탐색] 1차 분류기 그대로 층만 바꾼 T+A (판정에 쓰지 않음, 구간 없음)")
    for m, e in r["explore"].items():
        L.append(f"  {LABEL[m]:<8} 평균 AUC {e['auc_mean']:.4f} (T 대비 {e['auc_mean_diff']:+.4f}), "
                 f"평균 최대F1 {e['f1_mean']:.4f} (T 대비 {M[m]['f1_mean'] - M['LR_T']['f1_mean']:+.4f})")
    L.append("")
    d = r["decision"]
    L.append("[사전 등록 판정 (수정본, 1차 로지스틱 기준)]")
    L.append(f"  (a) 평균: T+A−T {dv('LR_T+A - LR_T'):+.4f} >= {GATE_MEAN_DIFF}, 하한 {B['LR_T+A - LR_T']['mean_ci'][0]:+.4f} > 0, "
             f"T+A−T_raw 하한 {B['LR_T+A - T_raw']['mean_ci'][0]:+.4f} > 0 : {'충족' if d['cond_a_mean'] else '불충족'}")
    L.append(f"  (b) 증상별(T+A−T >= {GATE_CLASS_DIFF}·하한>0, T+A−T_raw 하한>0, 대조군−T 구간이 0 포함) 통과: "
             f"{', '.join(d['cond_b_classes']) or '없음'}")
    L.append(f"  => 2단계(누적 예측 확장): {'간다' if d['go_stage2'] else '가지 않는다'}")
    L.append(f"  대조군−T 평균 {dv('LR_T+A_shuf - LR_T'):+.4f} {ci('LR_T+A_shuf - LR_T')}: "
             f"{'구간이 0 포함 (정상)' if d['control_mean_ci_contains_0'] else '구간이 0을 벗어남 — 확인 필요'}"
             + (f"; 증상별 구간이 0을 벗어난 증상: {', '.join(d['control_class_ci_excluding_0'])}"
                if d["control_class_ci_excluding_0"] else "; 증상별 구간도 모두 0 포함"))
    L.append(f"(소요 {r['seconds']:.0f}s)")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
