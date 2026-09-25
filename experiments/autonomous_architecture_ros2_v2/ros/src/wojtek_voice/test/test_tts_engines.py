

def test_build_engine_remote_is_url_only(monkeypatch):
    from wojtek_voice.tts_engines import RemoteEngine, build_engine

    engine = build_engine("remote", url="http://10.0.0.2:8120/")
    assert isinstance(engine, RemoteEngine)
    assert engine.url == "http://10.0.0.2:8120"
    monkeypatch.setenv("TTS_URL", "http://127.0.0.1:9999")
    assert build_engine("remote", url="").url == "http://127.0.0.1:9999"
