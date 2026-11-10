"""Optional HTTPS with a self-signed certificate, plus helpers for the room's network address.

A self-signed certificate encrypts the traffic between the dashboard and teachers' phones/laptops.
Browsers will show a one-time "not trusted" warning because no public authority signed it; accept it
once per device (or install server.crt on the devices). The certificate is regenerated automatically
when it is about to expire or when this computer's network address changes (e.g. a new DHCP address).
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path
from typing import Iterable, Set, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def primary_ip() -> str:
    """The address other devices on the network use to reach this computer."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # UDP "connect": no packet is actually sent
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _wanted(extra: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    dns, ips = {"localhost"}, {"127.0.0.1", primary_ip()}
    try:
        dns.add(socket.gethostname())
    except OSError:
        pass
    for e in extra:
        try:
            ips.add(str(ipaddress.ip_address(e)))
        except ValueError:
            if e:
                dns.add(e)
    return dns, ips


def _covers(cert: x509.Certificate, dns: Set[str], ips: Set[str]) -> bool:
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have_dns = set(san.get_values_for_type(x509.DNSName))
    have_ips = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    return dns <= have_dns and ips <= have_ips


def ensure_self_signed(folder: Path, extra_names: Iterable[str] = ()) -> Tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    crt, key = folder / "server.crt", folder / "server.key"
    dns, ips = _wanted(extra_names)
    if crt.exists() and key.exists():
        try:
            cert = x509.load_pem_x509_certificate(crt.read_bytes())
            fresh = cert.not_valid_after_utc > dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)
            if fresh and _covers(cert, dns, ips):
                return crt, key
        except ValueError:
            pass                                      # unreadable: make a new one

    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ExamGuard"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ExamGuard local server")])
    now = dt.datetime.now(dt.timezone.utc)
    san = [x509.DNSName(d) for d in sorted(dns)] + [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)]
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(private.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
                                         data_encipherment=False, key_agreement=True, key_cert_sign=True,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(private, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    return crt, key
