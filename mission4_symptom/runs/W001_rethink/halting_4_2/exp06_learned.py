"""exp06: 학습형 중단 정책을 4-2 목적함수에 맞춰 다시 만든다 (정책 개선, rollout).

E004 의 학습형은 "지금 끊는 것이 이후 어느 시점보다 낫다"(사후 최적, 순간 점수 기준)를 맞혔다.
여기서는 (1) 보상을 풀링 Macro-F1 의 1차 근사로 바꾸고  (2) 비교 대상을 사후 최적이 아니라
"한 발화 더 듣고 기준 정책(기대 점수 >= 0.52)을 따랐을 때"로 바꾼다.

  통화 i 를 t 에서 끊을 때의 효용  u_i(t) = E x Σ_c [TP_c A_c - FP_c B_c] + F x 조기판단력_i(t)
      A_c = N (2 - F1_c) / (9 D_c),  B_c = N F1_c / (9 D_c)   (FN->TP, FP 한 건이 Macro-F1 에 주는 변화)
  목표  δ_i(t) = u_i(t) - u_i(기준 정책이 t 뒤에 끊는 시점)
  정책  δ̂ >= θ 인 첫 순간에 중단 (5번째 이후).

특징: 확률 계열(FA) / + 대화 진행·길이 추정(FB) / + 멤버 불일치(FC, 모델 평균일 때).
평가: 1차는 통화 단위 5겹 교차 학습 + θ 를 4겹에서 골라 남은 1겹 채점. 가장 좋은 구성은 완전 중첩 교차검증으로 다시 잰다.
"""
from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np

import common as C
import feats as F
from m4 import decision as dec

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(C.MODELS)
NESTED = len(sys.argv) > 2 and sys.argv[2] == "nested"
rows = []
THETAS = np.round(np.arange(-0.12, 0.081, 0.01), 3)


def gbm():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=15, min_child_samples=100, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, random_state=0, verbose=-1)


def utilities(ctx, h0, p0, mask):
    """학습 통화(mask)의 기준 운영점에서 계산한 효용 u (순간별)."""
    idx = np.nonzero(mask)[0]
    tp, fp, fn = C.f1_counts(p0[idx], ctx.labels[idx])
    Fm, f1 = C.macro_f1_from(tp, fp, fn)
    D = np.maximum(2 * tp + fp + fn, 1)
    A, B = (2 - f1) / D * len(idx) / 9, f1 / D * len(idx) / 9
    E = float(C.earliness(h0[idx], ctx.n[idx]).mean())
    pr = ctx.pred_thr
    return E * ((pr & ctx.gold) @ A - (pr & ~ctx.gold) @ B) + float(Fm) * C.earliness(ctx.step, np.repeat(ctx.n, ctx.n))


def rollout_target(ctx, u, stop0):
    """δ(t) = u(t) - u(t 뒤에 기준 정책이 끊는 시점). 마지막 발화는 nan (계속할 수 없음)."""
    M = ctx.M
    nxt = np.full(M, -1, dtype=np.int64)          # t 뒤(> t)에서 기준 정책이 끊는 순간의 전역 인덱스
    for i in range(ctx.N):
        a, b = ctx.off[i], ctx.off[i + 1]
        cur = b - 1                                # 끝까지 못 끊으면 마지막 발화(finalize)
        for j in range(b - 1, a - 1, -1):
            nxt[j] = cur if j < b - 1 else -1
            if stop0[j] and ctx.step[j] >= C.GRACE:
                cur = j
    d = np.full(M, np.nan)
    ok = nxt >= 0
    d[ok] = u[ok] - u[nxt[ok]]
    return d, nxt


def crossfit(X, y, call_fold, rows_ok, folds):
    """folds 에 든 겹끼리 교차 학습한 예측 (그 밖의 겹은 건드리지 않는다)."""
    out = np.full(len(y), np.nan)
    for k in folds:
        tr = rows_ok & np.isin(call_fold, [f for f in folds if f != k])
        te = call_fold == k
        out[te] = gbm().fit(X[tr], y[tr]).predict(X[te])
    return out


for m in MODELS:
    ctx = C.Ctx(m)
    calls = C.load_dev_calls(ctx)
    base = C.baseline(ctx)
    base_fm = C.fold_mean(ctx, *base)
    v, cal, st = ctx.value, ctx.cal, ctx.step
    stop0 = v >= 0.52
    cf = ctx.fold[ctx.call_of]
    allm = np.ones(ctx.N, dtype=bool)
    print(f"\n==== {m}: 기준 {C.score_hp(*base, ctx.labels, ctx.n)['score']:.4f}, 겹평균 {base_fm:.4f}", flush=True)

    # ---- 특징
    srt = -np.sort(-cal, axis=1)[:, :3]
    pr = ctx.pred_thr
    same_prev = (pr == ctx.prev(pr)).all(axis=1) & ~ctx.first
    run_len = np.zeros(ctx.M)
    vmax = v.copy()
    for i in range(1, ctx.M):
        if not ctx.first[i]:
            run_len[i] = run_len[i - 1] + 1 if same_prev[i] else 0
            vmax[i] = max(vmax[i - 1], v[i])
    FA = np.column_stack([cal, srt, v, v - ctx.prev(v), vmax, run_len, cal.sum(axis=1), ctx.raw.max(axis=1), pr.sum(axis=1), st,
                          (cal - ctx.prev(cal)).max(axis=1), np.abs(cal - ctx.prev(cal)).sum(axis=1)])
    TX = F.build(calls)
    nhat = np.load(C.OUT / "dev_log_nhat.npy")
    FB = np.column_stack([FA, TX[:, 1:], nhat])
    sets = {"FA 확률 계열": FA, "FB +대화 진행·길이 추정": FB}
    if len(ctx.members) > 1:
        mv = np.stack([dec.expected_rule(dec.calibrate(x, C.SHIFT))[1] for x in ctx.members])
        mp = [dec.threshold_rule(x.copy()) for x in ctx.members]
        agree = np.all([(mp[0] == x).all(axis=1) for x in mp[1:]], axis=0).astype(float)
        sets["FC +멤버 불일치"] = np.column_stack([FB, mv.T, mv.std(axis=0), agree, np.std(ctx.members, axis=0).max(axis=1)])

    def policy_cands(score_arr, grid):
        out = {}
        for th in grid:
            h = C.first_stop(score_arr >= th, ctx.off)
            out[(float(th),)] = (h, C.take(ctx, h, pr))
        return out

    u_all = utilities(ctx, *base, allm)
    d_all, nxt = rollout_target(ctx, u_all, stop0)
    okn = nxt >= 0
    # 통화 끝의 보정 확률로 계산한 기대 효용 (R4), 기준 정책이 이미 끊은 뒤의 순간 표시 (R5)
    qend = np.repeat(cal[ctx.off[1:] - 1], ctx.n, axis=0)
    tp0, fp0, fn0 = C.f1_counts(base[1], ctx.labels)
    F0, f10 = C.macro_f1_from(tp0, fp0, fn0)
    D0 = 2 * tp0 + fp0 + fn0
    uq = float(C.earliness(base[0], ctx.n).mean()) * (pr * (qend * (2 - f10) / D0 - (1 - qend) * f10 / D0)).sum(axis=1) * ctx.N / 9         + float(F0) * C.earliness(st, np.repeat(ctx.n, ctx.n))
    stop_before = st > np.repeat(base[0], ctx.n) - 0     # 기준 정책의 중단 시점 이후(포함 안 함은 아래에서)
    stop_before = st >= np.repeat(base[0], ctx.n)
    ok = (st >= C.GRACE) & ~np.isnan(d_all)
    print(f"  목표 δ: 평균 {np.nanmean(d_all[ok]):+.4f}, δ>0(지금 끊는 편이 나음) 비율 {(d_all[ok] > 0).mean():.3f}, 기준 정책이 끊은 순간의 평균 δ {np.nanmean(d_all[ok & stop0]):+.4f}, "
          f"안 끊은 순간 {np.nanmean(d_all[ok & ~stop0]):+.4f}", flush=True)

    if not NESTED:
        for name, X in sets.items():
            t0 = time.time()
            dh = crossfit(X, np.nan_to_num(d_all), cf, ok, list(range(C.FOLDS)))
            r2 = 1 - np.nansum((d_all[ok] - dh[ok]) ** 2) / np.nansum((d_all[ok] - np.nanmean(d_all[ok])) ** 2)
            np.save(C.OUT / f"exp06_dhat_{m}_{name[:2]}.npy", dh.astype(np.float32))
            print(f"  [{name}] δ̂ 교차 학습 R2 {r2:.4f} ({time.time() - t0:.0f}초)", flush=True)
            rows.append(C.report(ctx, C.cv_select(ctx, policy_cands(dh, THETAS), f"R1 {name}: δ̂ >= θ"), base, base_fm))
            # 기대 점수와 섞기: 기대 점수 + β δ̂ >= τ
            cands = {}
            for beta in (0.25, 0.5, 1.0, 2.0):
                for t in np.round(np.arange(0.40, 0.65, 0.02), 2):
                    h = C.first_stop(v + beta * dh >= t, ctx.off)
                    cands[(beta, float(t))] = (h, C.take(ctx, h, pr))
            rows.append(C.report(ctx, C.cv_select(ctx, cands, f"R2 {name}: 기대점수 + β δ̂ >= τ"), base, base_fm))
            # R4. 분산을 줄인 목표: 정답 대신 통화 끝의 보정 확률 q 를 "사실상의 정답"으로 써서 기대 효용을 계산
            dq = np.full(ctx.M, np.nan)
            dq[okn] = uq[okn] - uq[nxt[okn]]
            dh4 = crossfit(X, np.nan_to_num(dq), cf, ok, list(range(C.FOLDS)))
            r2q = 1 - np.nansum((dq[ok] - dh4[ok]) ** 2) / np.nansum((dq[ok] - np.nanmean(dq[ok])) ** 2)
            print(f"  [{name}] 끝 확률 기준 목표의 교차 학습 R2 {r2q:.4f}", flush=True)
            rows.append(C.report(ctx, C.cv_select(ctx, policy_cands(dh4, THETAS), f"R4 {name}: 끝 확률 기준 δ̂ >= θ"), base, base_fm))
            cands = {}
            for beta in (0.5, 1.0, 2.0, 4.0):
                for t in np.round(np.arange(0.40, 0.65, 0.02), 2):
                    h = C.first_stop(v + beta * dh4 >= t, ctx.off)
                    cands[(beta, float(t))] = (h, C.take(ctx, h, pr))
            rows.append(C.report(ctx, C.cv_select(ctx, cands, f"R4' {name}: 기대점수 + β δ̂(끝 확률) >= τ"), base, base_fm))
            # R5. "더 들어도 예측이 안 바뀐다" 분류기: 기준 정책이 아직 안 끊은 순간에서, 지금 예측 == 기준 정책이 끊을 때의 예측
            same = np.zeros(ctx.M)
            same[okn] = (pr[okn] == pr[nxt[okn]]).all(axis=1)
            on = okn & (st >= C.GRACE) & ~stop_before        # 기준 정책이 아직 듣고 있는 순간만
            ph = np.full(ctx.M, np.nan)
            for k in range(C.FOLDS):
                tr, te = on & (cf != k), cf == k
                clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.06, num_leaves=15, min_child_samples=100, subsample=0.7,
                                         subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, random_state=0, verbose=-1)
                ph[te] = clf.fit(X[tr], same[tr]).predict_proba(X[te])[:, 1]
            from sklearn.metrics import roc_auc_score
            print(f"  [{name}] '안 바뀐다' 분류 AUC {roc_auc_score(same[on], ph[on]):.3f}, 양성 비율 {same[on].mean():.3f}", flush=True)
            cands = {}
            for t in np.round(np.arange(0.46, 0.59, 0.02), 2):
                for rho in (0.5, 0.6, 0.7, 0.8, 0.9, 9.0):
                    h = C.first_stop((v >= t) | (ph >= rho), ctx.off)
                    cands[(float(t), rho)] = (h, C.take(ctx, h, pr))
            rows.append(C.report(ctx, C.cv_select(ctx, cands, f"R5 {name}: 기대점수 >= τ OR P(안 바뀜) >= ρ"), base, base_fm))
        C.save_json(f"exp06_learned_{'_'.join(MODELS)}.json", rows)
        continue

    # ---- 완전 중첩 교차검증: 바깥 겹 k 의 통화는 δ 계산·학습·θ 선택 어디에도 쓰이지 않는다
    for name, X in sets.items():
        oof_h = np.empty(ctx.N, dtype=np.int64)
        cvs, picks = [], []
        for k in range(C.FOLDS):
            trc = ctx.fold != k
            u = utilities(ctx, *base, trc)
            d, _ = rollout_target(ctx, u, stop0)
            okk = (st >= C.GRACE) & ~np.isnan(d)
            inner = [f for f in range(C.FOLDS) if f != k]
            dh_in = crossfit(X, np.nan_to_num(d), cf, okk, inner)
            tr_idx = np.nonzero(trc)[0]
            sc = [C.score(ctx, C.first_stop(np.nan_to_num(dh_in, nan=-9) >= th, ctx.off), pr, tr_idx)["score"] for th in THETAS]
            th = float(THETAS[int(np.argmax(sc))])
            dh_te = gbm().fit(X[okk & (cf != k)], d[okk & (cf != k)]).predict(X)
            h = C.first_stop(dh_te >= th, ctx.off)
            te = np.nonzero(ctx.fold == k)[0]
            oof_h[te] = h[te]
            cvs.append(C.score(ctx, h, pr, te)["score"])
            picks.append(th)
            print(f"    겹 {k}: θ {th:+.2f}  학습 겹 점수 {max(sc):.4f}  남은 겹 {cvs[-1]:.4f}", flush=True)
        res = {"label": f"R1n {name}: δ̂ >= θ (완전 중첩)", "best_key": "-", "dev_all": float("nan"), "cv": float(np.mean(cvs)), "cv_folds": cvs,
               "picks": picks, "oof_h": oof_h, "oof_p": C.take(ctx, oof_h, pr)}
        rows.append(C.report(ctx, res, base, base_fm))
    C.save_json(f"exp06_nested_{'_'.join(MODELS)}.json", rows)
