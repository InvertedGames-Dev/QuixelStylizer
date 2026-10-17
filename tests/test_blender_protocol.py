import json

from qs_blender_server import handle_job


def test_ping_and_unknown():
    from qs.blender import EXPECTED_SERVER_VERSION
    ping = handle_job({"op": "ping"})
    assert ping["ok"] is True
    # The client restarts a warm server that reports any other version.
    assert ping["result"]["version"] == EXPECTED_SERVER_VERSION
    assert handle_job({"op": "quit"})["ok"] is True
    bad = handle_job({"op": "nope"})
    assert bad["ok"] is False
    json.dumps(bad)


def test_client_finds_script():
    from qs.blender import SERVER_SCRIPT, find_blender
    assert SERVER_SCRIPT.is_file()
    exe = find_blender()
    assert exe is None or exe.lower().endswith("blender.exe")
