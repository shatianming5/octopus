"""ask(): one streamed chat call to the Grok gateway (octopus/envs/swe/data/statement_coverage.py). The API comes
from GROK_MODELS_BASE_URL / XAI_API_KEY (environment only), the model from REWRITE_MODEL (grok-4.6). The branch/join
pool's acceptance-check rewrite (draft / apply) and its audit are at commit a6e83a9.
"""
import json, os, time, urllib.request


def ask(prompt, attempts=4):
    # streamed: a long reasoning turn sent whole outlives the gateway's idle timeout (502)
    body = json.dumps({"model": os.environ.get("REWRITE_MODEL", "grok-4.6"), "temperature": 0.3, "stream": True,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for i in range(attempts):
        try:
            req = urllib.request.Request(os.environ["GROK_MODELS_BASE_URL"].rstrip("/") + "/chat/completions", body, {
                "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['XAI_API_KEY']}",
                "User-Agent": "octopus-rewrite/1.0"})  # the gateway refuses urllib's default agent (502 forbidden)
            parts = []
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        for choice in json.loads(line[6:]).get("choices") or []:
                            parts.append((choice.get("delta") or {}).get("content") or "")
            return "".join(parts)
        except Exception as exc:  # transient gateway errors: back off and retry
            err = exc
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"API failed: {err}")
