from types import SimpleNamespace

import run


def test_flask_dotenv_loading_is_disabled(monkeypatch):
    cfg = SimpleNamespace(
        profile="laptop", role="standalone", https=False, host="127.0.0.1",
        public_url="", scheme="http", port=5000, tls_cert="", tls_key="",
    )
    monitor = object()
    svc = SimpleNamespace(
        start=lambda: None,
        stop=lambda: None,
        all_monitors=lambda: [monitor],
        notifier=SimpleNamespace(enabled=False),
    )
    app_options = {}

    class App:
        def run(self, **options):
            app_options.update(options)

    monkeypatch.setattr(run.Config, "from_env", lambda **_: cfg)
    monkeypatch.setattr(run, "build_services", lambda *_args, **_kwargs: svc)
    monkeypatch.setattr(run, "create_app", lambda _svc: App())
    monkeypatch.setattr(run, "ensure_password", lambda _cfg: "test-password")
    monkeypatch.setattr(run.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(run.sys, "argv", ["run.py"])
    monkeypatch.delenv("EXAMGUARD_SUPERVISED", raising=False)

    assert run.main() == 0
    assert app_options["load_dotenv"] is False
