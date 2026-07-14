"""Rewrites a Real VLA backend config's server_url/health_url to point
at a freshly (re)started Colab tunnel (ngrok/cloudflared), without
touching any other key.

RealVLAPolicyClient (policy/real_vla_policy_client.py) reads server_url/
health_url purely from the config file it's given -- no code change is
ever needed there, only the config. This script exists because a Colab
tunnel issues a new URL every time the notebook is (re)run, so editing
the config by hand every session would be tedious and error-prone.

Usage:
  python scripts/update_colab_vla_config.py \\
    --base-url https://xxxx.ngrok-free.app \\
    --config configs/real_vla_backend_colab_config.json
"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def validate_base_url(base_url: str) -> str:
    """Raises ValueError with a readable message for a missing scheme
    or host; otherwise returns base_url with any trailing slash(es)
    stripped."""
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"--base-url must start with http:// or https://, got: {base_url!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"--base-url must include a host, got: {base_url!r}")
    return base_url.rstrip("/")


def main() -> None:
    args = parse_args()

    try:
        base_url = validate_base_url(args.base_url)
    except ValueError as exc:
        print(str(exc))
        print("FAIL")
        return

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("FAIL")
        return

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    config["server_url"] = f"{base_url}/predict"
    config["health_url"] = f"{base_url}/health"

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")

    print(f"Updated {config_path}")
    print(f"server_url: {config['server_url']}")
    print(f"health_url: {config['health_url']}")
    print("PASS")


if __name__ == "__main__":
    main()
