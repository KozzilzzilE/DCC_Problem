"""Mission 3 제출 규격 검증 — inference.py 1회 실행으로 올바른 CSV 가 나오는가.

대회 실행 규칙:
    python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path {ckpt}
                        --output ./outputs/mission3.csv
    Mission 3 csv: [label file name], [symptom]   (symptom 은 "['두통', '복통']" String)
"""
import ast
import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSION3_DIR = REPO_ROOT / "mission3_symptom"

# Mission 2 전용 의존성. Mission 3 는 텍스트 과제라 이것들 없이도 채점이 끝나야 한다.
OTHER_MISSION_DEPENDENCIES = {"librosa", "torchvision", "torchaudio"}


def load_root_inference(blocked=()):
    """루트 inference.py 를 모듈로 적재한다. blocked 의 패키지는 없는 것처럼 만든다."""
    blocked = set(blocked)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "root_inference_under_test", REPO_ROOT / "inference.py")
    module = importlib.util.module_from_spec(spec)

    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = real_import
    return module


def test_imports_without_other_mission_dependencies():
    """librosa / torchvision 이 없어도 스크립트가 적재돼야 한다.

    최상단에서 import 하면 Mission 3 채점에서도 import 단계에서 죽어 0점이 된다.
    torchvision 은 requirements 에 +cu118 로 고정돼 있어 CPU 환경 설치 실패가 현실적이다.
    """
    module = load_root_inference(blocked=OTHER_MISSION_DEPENDENCIES)
    assert hasattr(module, "mission3_inference")


def test_mission3_path_is_not_a_stub():
    """더미 반환으로 되돌아가는 회귀를 막는다."""
    source = (REPO_ROOT / "inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "mission3_inference"
    )
    body = ast.get_source_segment(source, func) or ""
    assert "sample1.json" not in body, "mission3_inference 가 더미 스텁으로 되돌아갔다"
    assert "predict_directory" in body, "m3 추론 경로를 호출하지 않는다"


def test_mission3_output_columns_match_specification():
    sys.path.insert(0, str(MISSION3_DIR))
    try:
        from m3.infer import OUTPUT_COLUMNS
    finally:
        sys.path.remove(str(MISSION3_DIR))
    assert OUTPUT_COLUMNS == ["label file name", "symptom"]


def test_decision_threshold_is_fixed_by_contest_rule():
    """대회 규정: 결정 임계값 0.5 고정. 저장된 보정값이 끼어들 자리가 없어야 한다."""
    sys.path.insert(0, str(MISSION3_DIR))
    try:
        from m3.infer import DECISION_THRESHOLD, decision_threshold
    finally:
        sys.path.remove(str(MISSION3_DIR))
    assert DECISION_THRESHOLD == 0.5
    assert decision_threshold() == 0.5


@pytest.mark.parametrize("dependency", sorted(OTHER_MISSION_DEPENDENCIES))
def test_dependency_is_not_imported_at_module_level(dependency):
    """실제 사용처 바깥에서 import 하면 다시 같은 사고가 난다."""
    source = (REPO_ROOT / "inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:                      # 모듈 최상단만 본다
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert dependency not in names, (
            f"{dependency} 가 최상단에서 import 되고 있다. 실제 쓰는 함수 안으로 내려야 한다"
        )
