# Mission 2 실험 실행 대응표

| 실험 흐름 | 실행 진입점 | 산출물 |
|---|---|---|
| ResNet 기준선·3모델 10에포크 | train_models.py --model resnet50 / all | last_*.pt, history.json, run_config.json |
| 입력 조건 네 가지 | train_models.py --window-suite | 조건별 가중치·학습 경과 |
| 200통화 탐색 | quick_hypotheses.py prepare/audit/probe, default Mel | 음향 통계·소거·모델별 대조 표 |
| 500통화 비교 | quick_hypotheses.py prepare/probe, resnet128 | seed별 comparison/differences CSV |
| 통합 실험 | run_all_audio_hypotheses.py --plan ... --stage ... | 조건별 지표·판단 |
| 신규 1,000통화 ZCR | run_zcr_confirmation.py --stage train / report | ZCR 비교·decision_summary.json |
| 단일·균등 7조합 | evaluate_models.py / m2.experiment_workflow.compare_ensembles | 확률·Accuracy·F1·Recall·혼동행렬 |
| 제출 추론 | ../inference.py | UTF-8 BOM 4열 CSV |

노트북 기본값은 결과 확인입니다. 학습은 실행 단계를 명시해 시작하며, 상세 명령과 가중치 배치는 [상세 실행 가이드](EXECUTION_GUIDE.md)를 참고하세요.
