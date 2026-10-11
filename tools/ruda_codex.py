"""Run the Codex fork against the exact model served by RUDA (no build)."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", required=True, type=Path,
                        help="Path to the built shuqi2077/codex fork executable")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="Exact served model name; defaults to the single loaded model")
    parser.add_argument("codex_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    executable = args.codex.resolve(strict=True)
    if not executable.is_file():
        parser.error("--codex must name an executable file")
    base_url = args.base_url.rstrip("/")
    request = Request(f"{base_url}/codex/models", headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        catalog = json.load(response)
    models = catalog.get("models", [])
    if not models:
        parser.error("RUDA returned no model catalog; enable --responses-max-output-tokens on ruda-serve")
    model = args.model
    if model is None:
        if len(models) != 1:
            parser.error("--model is required when multiple models are served")
        model = models[0]["slug"]
    if not any(item.get("slug") == model for item in models):
        parser.error("--model must exactly match the RUDA catalog")
    settings = {
        "model": model,
        "model_provider": "ruda",
        "web_search": "disabled",
        "model_reasoning_summary": "none",
        "model_reasoning_effort": "none",
        "tools.update_plan.enabled": True,
        "model_providers.ruda.name": "RUDA",
        "model_providers.ruda.base_url": base_url,
        "model_providers.ruda.model_catalog_url": f"{base_url}/codex/models",
        "model_providers.ruda.wire_api": "responses",
        "model_providers.ruda.requires_openai_auth": False,
        "model_providers.ruda.supports_websockets": False,
        "model_providers.ruda.capabilities.remote_compaction": "unsupported",
        "model_providers.ruda.capabilities.external_web_access": False,
    }
    extra = args.codex_args[1:] if args.codex_args[:1] == ["--"] else args.codex_args
    with tempfile.TemporaryDirectory(prefix="ruda-codex-") as directory:
        catalog_file = Path(directory) / "models.json"
        catalog_file.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        settings["model_catalog_json"] = str(catalog_file)
        command = [str(executable)]
        for key, value in settings.items():
            command.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        # Codex retains the caller's working directory, approval and sandbox policies.
        return subprocess.call(command + extra)


if __name__ == "__main__":
    raise SystemExit(main())
