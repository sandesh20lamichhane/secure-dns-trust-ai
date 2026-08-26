"""Live TLS handshake probe: fetch the leaf certificate for a domain.

Records observations, not judgements. Every row carries observed_at, because a
certificate is a point-in-time fact - this is the reason data/01_collected/ is
write-once rather than regenerable.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as _dt
import ssl
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from .ledger import Ledger
from ..utils.io import ShardWriter

FREE_CA_MARKERS = ("let's encrypt", "lets encrypt", "zerossl", "buypass",
                   "ssl.com free", "cpanel", "actalis free")


def _permissive_context() -> ssl.SSLContext:
    """We must retrieve invalid and expired certificates - those are signal, not
    error. Verification is performed later, in the feature layer."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    return ctx


def parse_certificate(der: bytes, domain: str, observed_at: str) -> dict:
    cert = x509.load_der_x509_certificate(der)

    def _attr(name, oid):
        try:
            vals = name.get_attributes_for_oid(oid)
            return vals[0].value if vals else None
        except Exception:
            return None

    from cryptography.x509.oid import ExtensionOID, NameOID

    subject_cn = _attr(cert.subject, NameOID.COMMON_NAME)
    issuer_cn = _attr(cert.issuer, NameOID.COMMON_NAME)
    issuer_org = _attr(cert.issuer, NameOID.ORGANIZATION_NAME)

    try:
        san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        sans = san_ext.value.get_values_for_type(x509.DNSName)
    except Exception:
        sans = []

    not_before = cert.not_valid_before_utc
    not_after = cert.not_valid_after_utc
    now = _dt.datetime.now(_dt.timezone.utc)

    pub = cert.public_key()
    try:
        key_bits = pub.key_size
    except Exception:
        key_bits = None

    return {
        "domain": domain,
        "observed_at": observed_at,
        "subject_cn": subject_cn,
        "issuer_cn": issuer_cn,
        "issuer_org": issuer_org,
        "is_free_ca": bool(issuer_org and any(m in issuer_org.lower() for m in FREE_CA_MARKERS)),
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "validity_days": (not_after - not_before).days,
        "days_until_expiry": (not_after - now).days,
        "cert_age_days": (now - not_before).days,
        "is_expired": now > not_after,
        "is_not_yet_valid": now < not_before,
        "is_self_signed": cert.issuer == cert.subject,
        "san_count": len(sans),
        "san_list": ",".join(sans[:50]),
        "wildcard_san": any(s.startswith("*.") for s in sans),
        "cn_in_san": bool(subject_cn and subject_cn in sans),
        "key_algorithm": type(pub).__name__.replace("_", "").replace("PublicKey", ""),
        "key_bits": key_bits,
        "sig_algorithm": getattr(cert.signature_algorithm_oid, "_name", None),
        "serial_number": str(cert.serial_number),
        "version": cert.version.name,
        "sha256_fingerprint": cert.fingerprint(hashes.SHA256()).hex(),
        # TLSA selector-1 / matching-type-1 payload: SHA-256 of SubjectPublicKeyInfo.
        # Precomputed here so tlsa_matches_cert is a cheap lookup later.
        "spki_sha256": hashes.Hash(hashes.SHA256()).__class__ and _spki_sha256(pub),
    }


def _spki_sha256(pub) -> str:
    der = pub.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    digest = hashes.Hash(hashes.SHA256())
    digest.update(der)
    return digest.finalize().hex()


async def probe_one(domain: str, port: int = 443, timeout: float = 10.0) -> tuple:
    """Return (domain, status, row_or_None, error_or_None)."""
    observed_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        ctx = _permissive_context()
        fut = asyncio.open_connection(domain, port, ssl=ctx, server_hostname=domain)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout)
        try:
            der = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if not der:
            return domain, "no_certificate", None, None
        return domain, "success", parse_certificate(der, domain, observed_at), None
    except asyncio.TimeoutError:
        return domain, "timeout", None, "handshake timeout"
    except ssl.SSLError as exc:
        return domain, "tls_error", None, str(exc)[:200]
    except (OSError, ConnectionError) as exc:
        msg = str(exc)
        status = "dns_error" if "Name or service not known" in msg or "getaddrinfo" in msg \
            else "connection_error"
        return domain, status, None, msg[:200]
    except Exception as exc:                       # malformed cert, unusual encoding
        return domain, "tls_error", None, f"{type(exc).__name__}: {str(exc)[:180]}"


def _run(coro):
    """Execute a coroutine whether or not an event loop is already running.

    Colab and Jupyter run the kernel inside an asyncio loop, so asyncio.run()
    raises "cannot be called from a running event loop". Rather than depend on
    nest_asyncio, the coroutine is handed to a worker thread that owns a fresh
    loop - correct in both a plain script and a notebook.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def probe_batch(domains: Iterable[str], concurrency: int = 100,
                      timeout: float = 10.0) -> list[tuple]:
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(d):
        async with sem:
            return await probe_one(d, timeout=timeout)

    return await asyncio.gather(*(_guarded(d) for d in domains))


def run_collection(ledger: Ledger, out_dir, prefix: str = "tls_probe",
                   batch_size: int = 2_000, concurrency: int = 100,
                   max_batches: int | None = None, logger=None) -> dict:
    """Resumable driver. Safe to interrupt and re-run: the ledger is the state."""
    writer = ShardWriter(out_dir, prefix)
    batches = 0
    try:
        while True:
            todo = ledger.pending(limit=batch_size)
            if not todo:
                break
            if max_batches is not None and batches >= max_batches:
                break
            results = _run(probe_batch(todo, concurrency=concurrency))
            for domain, status, row, error in results:
                if row:
                    writer.add(row)
                ledger.mark(domain, status, shard_id=writer._shard_id, error=error)
            ledger.commit(backup=True)
            batches += 1
            if logger:
                logger.info("batch %d done | %s", batches, ledger.summary())
    finally:
        writer.flush()
        ledger.commit(backup=True)
    return ledger.summary()
