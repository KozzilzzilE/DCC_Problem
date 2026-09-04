"""콘솔 출력 인코딩 보정.

한국어 Windows 의 기본 콘솔 인코딩은 cp949 다. 한글은 담기지만 em-dash(—)나
이모지 같은 문자는 없어서, 진행 메시지 한 줄 때문에 UnicodeEncodeError 로
스크립트 전체가 죽는다. 결과 파일을 다 쓰고 나서 출력 단계에서 죽으면 종료
코드만 1 이 되어 실패로 오인되기 쉽다.
"""
from __future__ import annotations

import sys


def ensure_utf8_stdout() -> None:
    """stdout/stderr 을 UTF-8 로 고정한다. CLI 진입점에서 가장 먼저 호출한다."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # 이미 닫혔거나 재설정할 수 없는 스트림
