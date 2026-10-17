import io
import zipfile

import pytest

from qs.core.extract import extract_zip, safe_extract


def test_safe_extract_rejects_escape(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "no")
    buf.seek(0)
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ValueError):
            safe_extract(zf, str(dest))


def test_extract_ok(tmp_path):
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("maps/Albedo.png", b"x")
    dest = tmp_path / "src"
    extract_zip(str(zpath), str(dest))
    assert (dest / "maps" / "Albedo.png").is_file()
