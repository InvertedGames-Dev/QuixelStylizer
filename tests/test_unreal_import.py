from tools.unreal.import_stylized import _content_path, _suffix


def test_suffixes():
    assert _suffix("Rock_D.png") == "D"
    assert _suffix("Rock_MROD.png") == "MROD"
    assert _suffix("Rock_N.tga") == "N"
    assert _suffix("notes.txt") is None


def test_content_path():
    p = _content_path(r"C:\Users\BigMoney\Projects\Exodus\Content\Mats\Rock")
    assert p == "/Game/Mats/Rock"
