"""DNS record and DNSSEC/TLSA collection (dnspython).

TLSA lookup is the piece that ties this study back to the SATAF framework:
it establishes how many domains actually publish TLSA records, which is the
empirical motivation for a probabilistic trust score in the first place.
"""
from __future__ import annotations

import datetime as _dt

import dns.flags
import dns.rdatatype
import dns.resolver


def _resolver(nameservers=None, timeout: float = 5.0) -> dns.resolver.Resolver:
    r = dns.resolver.Resolver()
    if nameservers:
        r.nameservers = nameservers
    r.timeout = timeout
    r.lifetime = timeout
    return r


def collect(domain: str, resolver=None, port: int = 443, protocol: str = "tcp") -> dict:
    r = resolver or _resolver()
    row = {"domain": domain,
           "observed_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}

    def _q(name, rdtype, want_dnssec=False):
        try:
            return r.resolve(name, rdtype, raise_on_no_answer=False)
        except Exception:
            return None

    a = _q(domain, "A")
    row["n_a_records"] = len(a) if a and a.rrset else 0
    row["ttl_a"] = a.rrset.ttl if a and a.rrset else None

    aaaa = _q(domain, "AAAA")
    row["has_aaaa"] = bool(aaaa and aaaa.rrset)

    ns = _q(domain, "NS")
    row["n_ns_records"] = len(ns) if ns and ns.rrset else 0
    row["ns_diversity"] = len({str(x).split(".", 1)[-1] for x in ns}) if ns and ns.rrset else 0

    mx = _q(domain, "MX")
    row["has_mx"] = bool(mx and mx.rrset)

    # DNSSEC: the AD flag from a validating resolver
    try:
        resp = r.resolve(domain, "A", raise_on_no_answer=False).response
        row["dnssec_signed"] = bool(resp.flags & dns.flags.AD)
    except Exception:
        row["dnssec_signed"] = False

    # TLSA - the DANE record
    tlsa_name = f"_{port}._{protocol}.{domain}"
    try:
        ans = r.resolve(tlsa_name, "TLSA", raise_on_no_answer=False)
        if ans and ans.rrset:
            rec = ans.rrset[0]
            row.update({
                "has_tlsa": True,
                "tlsa_usage": rec.usage,
                "tlsa_selector": rec.selector,
                "tlsa_matching_type": rec.mtype,
                "tlsa_cert_association": rec.cert.hex(),
                "tlsa_record_count": len(ans.rrset),
                "tlsa_response_bytes": len(ans.response.to_wire()),
            })
        else:
            row["has_tlsa"] = False
    except Exception:
        row["has_tlsa"] = False
    return row
