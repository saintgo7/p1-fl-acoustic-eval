# ElectricityMaps API로 한국 전력망 탄소강도를 조회해 P3 에너지→탄소 환산을 지원하는 모듈
#
# =============================================================================
# API 조사 결과 (2026-06-03, WebSearch 검증)
# =============================================================================
# 엔드포인트  : GET https://api.electricitymap.org/v3/carbon-intensity/latest?zone=KR
# 인증 헤더   : auth-token: <token>
# 응답 필드   : zone, carbonIntensity (gCO2eq/kWh), datetime, updatedAt,
#               emissionFactorType, isEstimated, estimationMethod
# 무료 플랜   : 라이브(현재) 데이터만 제공, 과거 이력(history) 불가
#               단일 zone 제한, 시간당 50 요청 상한
# 참고 출처   :
#   https://portal.electricitymaps.com/docs/api
#   https://www.electricitymaps.com/free-tier-api
#   https://github.com/thegreenwebfoundation/grid-aware-websites/issues/21
# =============================================================================

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_API_BASE = "https://api.electricitymap.org/v3/carbon-intensity/latest"
_DEFAULT_ZONE = "KR"
_ENV_KEY = "ELECTRICITYMAPS_TOKEN"
_DOTENV_PATH = Path.home() / "abada-night" / ".env"
_TIMEOUT_SEC = 10


# ---------------------------------------------------------------------------
# 토큰 로딩
# ---------------------------------------------------------------------------
def load_token() -> str | None:
    """환경변수 또는 ~/abada-night/.env 에서 ELECTRICITYMAPS_TOKEN을 읽는다.

    두 곳 모두 없으면 None을 반환한다(호출자가 degraded 모드로 처리).
    """
    # 1순위: 환경변수
    token = os.environ.get(_ENV_KEY)
    if token:
        return token

    # 2순위: .env 파일 (KEY=VALUE 형식)
    if _DOTENV_PATH.exists():
        with open(_DOTENV_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() == _ENV_KEY:
                    return val.strip()

    return None


# ---------------------------------------------------------------------------
# 탄소강도 조회
# ---------------------------------------------------------------------------
def get_carbon_intensity(
    zone: str = _DEFAULT_ZONE,
    token: str | None = None,
) -> dict[str, Any]:
    """ElectricityMaps API에서 지정 zone의 현재 탄소강도를 조회한다.

    성공 시:
        {
            "carbonIntensity_gco2_kwh": float,   # gCO2eq/kWh
            "datetime": str,                     # ISO8601
            "zone": str,
            "isEstimated": bool,
            "estimationMethod": str | None,
            "source": "ElectricityMaps-API-v3",
            "provenance": "fetched-from-api",    # §11.2 출처 표시
        }
    실패 시:
        {"error": str, "provenance": "fetched-from-api"}
    """
    if token is None:
        return {
            "error": "ELECTRICITYMAPS_TOKEN not set — carbon_unavailable",
            "provenance": "fetched-from-api",
        }

    url = f"{_API_BASE}?zone={zone}"
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        req = urllib.request.Request(url, headers={"auth-token": token})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.reason}", "provenance": "fetched-from-api"}
    except urllib.error.URLError as exc:
        return {"error": f"URLError: {exc.reason}", "provenance": "fetched-from-api"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected: {exc}", "provenance": "fetched-from-api"}

    intensity = data.get("carbonIntensity")
    if intensity is None:
        return {"error": f"carbonIntensity field missing: {data}", "provenance": "fetched-from-api"}

    return {
        "carbonIntensity_gco2_kwh": float(intensity),
        "datetime": data.get("datetime"),
        "zone": data.get("zone", zone),
        "isEstimated": data.get("isEstimated", False),
        "estimationMethod": data.get("estimationMethod"),
        "source": "ElectricityMaps-API-v3",
        "provenance": "fetched-from-api",
    }


# ---------------------------------------------------------------------------
# 에너지 → 탄소 환산 (순수 함수)
# ---------------------------------------------------------------------------
def energy_to_carbon(energy_kwh: float, intensity_gco2_kwh: float) -> float:
    """측정 에너지(kWh)와 API 탄소강도(gCO2eq/kWh)로 CO2 배출량(g)을 계산한다.

    provenance:
        energy_kwh        — measured (NVML pynvml)
        intensity_gco2_kwh — fetched-from-api (ElectricityMaps)
        return value       — computed (product of above)
    """
    return energy_kwh * intensity_gco2_kwh


# ---------------------------------------------------------------------------
# W&B 로깅 헬퍼
# ---------------------------------------------------------------------------
def sample_carbon_to_wandb(run: Any, zone: str = _DEFAULT_ZONE) -> dict[str, Any]:
    """현재 탄소강도를 조회해 W&B run에 로깅한다.

    run.log()가 없으면 조용히 건너뛴다. 탄소강도 조회 실패 시에는
    'carbon_unavailable=1' 을 로깅해 시계열 공백을 표시한다.

    반환값: get_carbon_intensity() 결과 dict (호출자 감사용).
    """
    token = load_token()
    result = get_carbon_intensity(zone=zone, token=token)

    if not callable(getattr(run, "log", None)):
        return result

    if "error" in result:
        run.log({"carbon_unavailable": 1})
    else:
        run.log({
            "carbon_intensity_gco2_kwh": result["carbonIntensity_gco2_kwh"],
            "carbon_zone": result["zone"],
            "carbon_is_estimated": int(result["isEstimated"]),
        })

    return result


# ---------------------------------------------------------------------------
# 셀프 테스트 (python3 carbon.py 로 직접 실행 시)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== carbon.py self-test ===\n")

    # (a) load_token: 맥북에는 토큰 없음 → None
    token = load_token()
    print(f"[a] load_token() → {token!r}")
    assert token is None or isinstance(token, str), "load_token must return str|None"
    print("    PASS: returns None or str, no crash\n")

    # (b) energy_to_carbon: 순수 함수 검증
    result_g = energy_to_carbon(1.5, 400.0)
    print(f"[b] energy_to_carbon(1.5 kWh, 400 gCO2/kWh) → {result_g} g")
    assert result_g == 600.0, f"Expected 600.0, got {result_g}"
    print("    PASS: 600.0 g\n")

    # (c) get_carbon_intensity: 토큰 없으면 graceful error dict 반환
    intensity_result = get_carbon_intensity(zone="KR", token=token)
    print(f"[c] get_carbon_intensity(zone='KR', token={token!r})")
    print(f"    result → {intensity_result}")
    assert "provenance" in intensity_result, "provenance field missing"
    assert intensity_result.get("provenance") == "fetched-from-api", "provenance mismatch"
    if token is None:
        assert "error" in intensity_result, "Should return error dict when token is None"
        print("    PASS: graceful error (no token), no crash\n")
    else:
        print("    INFO: token found, result above\n")

    print("=== all self-tests passed ===")
