"""Write side of the content-addressed store: a stdlib S3 client with Signature V4.

It can put and head objects. It has no way to delete one.
"""

import hashlib
import hmac
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from bakeoff.store import TIMEOUT, GaveUp, StoreError, retry_call

REGION = "auto"
SERVICE = "s3"
DEFAULT_BUCKET = "bakeoff-data"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
UPLOAD_HEADERS = {
    "Content-Type": "application/octet-stream",
    "Cache-Control": "public, max-age=31536000, immutable",
}


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sign(
    method: str,
    url: str,
    headers: dict[str, str],
    payload_sha256: str,
    *,
    key_id: str,
    secret: str,
    region: str,
    service: str,
    amz_date: str,
) -> dict[str, str]:
    parts = urllib.parse.urlsplit(url)
    signed = {k.lower(): " ".join(v.split()) for k, v in headers.items()}
    signed["host"] = parts.netloc
    signed["x-amz-date"] = amz_date
    signed["x-amz-content-sha256"] = payload_sha256
    names = sorted(signed)
    path = "/".join(urllib.parse.quote(seg, safe="-_.~") for seg in (parts.path or "/").split("/"))
    query = "&".join(
        sorted(
            urllib.parse.quote(k, safe="-_.~") + "=" + urllib.parse.quote(v, safe="-_.~")
            for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        )
    )
    canonical = "\n".join(
        [
            method,
            path,
            query,
            "".join(f"{n}:{signed[n]}\n" for n in names),
            ";".join(names),
            payload_sha256,
        ]
    )
    day = amz_date[:8]
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    k = _hmac(("AWS4" + secret).encode(), day)
    for part in (region, service, "aws4_request"):
        k = _hmac(k, part)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    out = dict(headers)
    out["x-amz-date"] = amz_date
    out["x-amz-content-sha256"] = payload_sha256
    out["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return out


@dataclass
class S3Client:
    endpoint: str
    bucket: str
    key_id: str
    secret: str = field(repr=False)

    @classmethod
    def from_env(cls) -> "S3Client":
        values = {}
        for name in ("ENDPOINT", "KEY_ID", "SECRET"):
            var = f"BAKEOFF_STORE_{name}"
            values[name] = os.environ.get(var)
            if not values[name]:
                raise StoreError(f"{var} is not set")
        bucket = os.environ.get("BAKEOFF_STORE_BUCKET") or DEFAULT_BUCKET
        return cls(values["ENDPOINT"], bucket, values["KEY_ID"], values["SECRET"])

    def _send(self, method: str, key: str, sha256: str, path: Path | None) -> int:
        url = f"{self.endpoint.rstrip('/')}/{self.bucket}/{key}"
        extra = dict(UPLOAD_HEADERS) if path else {}

        def attempt() -> int:
            amz_date = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            headers = sign(
                method,
                url,
                extra,
                sha256,
                key_id=self.key_id,
                secret=self.secret,
                region=REGION,
                service=SERVICE,
                amz_date=amz_date,
            )
            body = path.open("rb") if path else None
            if path:
                headers["Content-Length"] = str(path.stat().st_size)
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    return resp.status
            except urllib.error.HTTPError as e:
                if e.code == 404 and method == "HEAD":
                    return 404
                raise
            finally:
                if body:
                    body.close()

        try:
            return retry_call(attempt)
        except urllib.error.HTTPError as e:
            raise StoreError(f"{method} {key}: status {e.code}") from None
        except GaveUp as e:
            raise StoreError(f"{method} {key}: gave up after retries (last {e.last})") from None

    def head(self, key: str) -> bool:
        return self._send("HEAD", key, EMPTY_SHA, None) == 200

    def put(self, key: str, path: Path, sha256: str) -> None:
        self._send("PUT", key, sha256, path)
