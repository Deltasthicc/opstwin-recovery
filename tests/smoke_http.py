"""
HTTP smoke test for the OpsTwin FastAPI server.

Probes the OpenEnv endpoints a HuggingFace Space reviewer (or OpenEnv client)
would hit: /health, /reset, and a couple of /step calls. Exits non-zero on
the first failure so this can gate deployment.

Usage:
    python tests/smoke_http.py [--base-url http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url: str, body: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    print(f"Smoke-testing {base}")

    # 1. Health
    try:
        h = _get(f"{base}/health")
        assert h.get("status") == "healthy", h
        print("  /health ........... OK")
    except (urllib.error.URLError, AssertionError) as e:
        print(f"  /health ........... FAIL ({e})")
        return 1

    # 2. Reset
    try:
        obs = _post(f"{base}/reset", {})
        assert "observation" in obs, obs
        o = obs["observation"]
        assert o.get("active_desk") is not None
        assert isinstance(o.get("available_commands"), list) and len(o["available_commands"]) > 0
        total = o.get("total_issues_count", 0)
        assert total > 0, f"scenario should load with issues > 0 (got {total})"
        print(f"  /reset ............ OK  desk={o['active_desk']}  "
              f"services={len(o.get('services', []))}  tickets={len(o.get('tickets', []))}  "
              f"total_issues={total}")
    except Exception as e:
        print(f"  /reset ............ FAIL ({e})")
        return 1

    # 3. Step: a safe read-only action that every desk accepts.
    #    A single REQUEST_INFO must NOT terminate the episode. If done=True
    #    here, state is not being preserved between /reset and /step.
    try:
        resp = _post(f"{base}/step", {"action": {"command": "REQUEST_INFO summary"}})
        assert "observation" in resp, resp
        assert isinstance(resp.get("reward"), (int, float)), resp.get("reward")
        assert isinstance(resp.get("done"), bool), resp.get("done")
        assert resp["done"] is False, \
            "one REQUEST_INFO terminated the episode; env state is not persisted across HTTP calls"
        print(f"  /step REQUEST_INFO  OK  reward={resp['reward']:.4f}  done={resp['done']}")
    except Exception as e:
        print(f"  /step REQUEST_INFO  FAIL ({e})")
        return 1

    # 4. Step: desk switch is always legal from any starting desk.
    try:
        resp = _post(f"{base}/step", {"action": {"command": "SWITCH_DESK SRE"}})
        r_obs = resp["observation"]
        assert r_obs.get("active_desk") == "SRE", r_obs.get("active_desk")
        print(f"  /step SWITCH_DESK   OK  active_desk={r_obs['active_desk']}")
    except Exception as e:
        print(f"  /step SWITCH_DESK   FAIL ({e})")
        return 1

    # 5. DONE terminates the episode cleanly.
    try:
        resp = _post(f"{base}/step", {"action": {"command": "DONE"}})
        assert resp.get("done") is True, resp
        final_score = resp["observation"].get("score")
        print(f"  /step DONE          OK  final_score={final_score}")
    except Exception as e:
        print(f"  /step DONE          FAIL ({e})")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
