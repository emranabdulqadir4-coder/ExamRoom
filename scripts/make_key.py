#!/usr/bin/env python3
"""Print a new secret key. Used for EXAMGUARD_MQTT_KEY (edge box <-> server) or EXAMGUARD_VAULT_KEY.

    python scripts/make_key.py

Put the SAME value in .env on both computers (EXAMGUARD_MQTT_KEY=...). Keep it secret: anyone with it and
access to the broker could read or forge the messages.
"""
from cryptography.fernet import Fernet

print(Fernet.generate_key().decode())
