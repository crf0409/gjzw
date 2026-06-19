#!/usr/bin/env python
"""Upload a release package to Volcengine TOS and print a presigned URL.

Credentials are read from environment variables documented in
``/home/siton02/disk_sdg/md0_backup_2026-04-28/crf/tos.md``. The script avoids
printing AK/SK values.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "release_packages"
DEFAULT_ENDPOINT = "https://tos-s3-cn-guangzhou.volces.com"
DEFAULT_REGION = "cn-guangzhou"
DEFAULT_BUCKET = "dbz"
DEFAULT_EXPIRES = 3 * 24 * 3600
TOS_ENV_KEYS = {
    "GMXLAB_STORAGE_ENDPOINT",
    "GMXLAB_STORAGE_REGION",
    "GMXLAB_STORAGE_BUCKET",
    "GMXLAB_STORAGE_AK",
    "GMXLAB_STORAGE_SK",
}


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def clean_doc_value(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and "`" in value[1:]:
        value = value[1:].split("`", 1)[0]
    return value.strip()


def load_env_from_config_doc(path: Path) -> None:
    """Load TOS env vars from the markdown table without printing secrets."""
    if not path.exists():
        raise SystemExit(f"missing config doc: {path}")
    loaded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = clean_doc_value(cells[0])
        if key not in TOS_ENV_KEYS:
            continue
        value = clean_doc_value(cells[1])
        if value and key not in os.environ:
            os.environ[key] = value
        loaded.add(key)
    missing = sorted(key for key in TOS_ENV_KEYS if not os.environ.get(key))
    if missing:
        raise SystemExit(f"config doc did not provide required variables: {', '.join(missing)}")
    print(f"loaded TOS config keys from: {path}")
    print("loaded keys: " + ", ".join(sorted(loaded)))


def build_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise SystemExit("boto3 is required for TOS upload") from exc

    endpoint = env("GMXLAB_STORAGE_ENDPOINT", DEFAULT_ENDPOINT)
    region = env("GMXLAB_STORAGE_REGION", DEFAULT_REGION)
    ak = env("GMXLAB_STORAGE_AK")
    sk = env("GMXLAB_STORAGE_SK")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", help="Local ZIP package path.")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--key", default=None, help="TOS object key. Defaults to releases/aafnet/<filename>.")
    parser.add_argument("--expires", type=int, default=DEFAULT_EXPIRES, help="Presigned URL expiry in seconds.")
    parser.add_argument("--expires-days", type=float, default=None, help="Presigned URL expiry in days.")
    parser.add_argument("--config-doc", default=None, help="Markdown config doc containing GMXLAB_STORAGE_* values.")
    args = parser.parse_args()

    if args.config_doc:
        load_env_from_config_doc(Path(args.config_doc).expanduser().resolve())

    package = Path(args.package).resolve()
    if not package.exists() or not package.is_file():
        raise SystemExit(f"missing package: {package}")
    bucket = args.bucket or os.environ.get("GMXLAB_STORAGE_BUCKET", DEFAULT_BUCKET)
    key = args.key or f"releases/aafnet/{package.name}"
    expires = int(args.expires_days * 24 * 3600) if args.expires_days is not None else args.expires

    client = build_client()
    client.upload_file(str(package), bucket, key)
    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_epoch": time.time(),
        "package": str(package),
        "bucket": bucket,
        "key": key,
        "expires": expires,
        "url": url,
    }
    report_path = OUT_DIR / f"{package.stem}.tos_upload.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"uploaded: s3://{bucket}/{key}")
    print(f"expires_sec: {expires}")
    print(f"url: {url}")
    print(f"report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
