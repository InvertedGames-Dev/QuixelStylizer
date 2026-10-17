from qs.core.maps import classify_file, detect_maps


def test_classify_quixel_names():
    assert classify_file("Rock_Cliff_tg1mcgbl_2K_Albedo.png")[0] == "albedo"
    assert classify_file("Rock_Cliff_tg1mcgbl_2K_NormalDX.png")[0] == "normal_dx"
    assert classify_file("Rock_Cliff_tg1mcgbl_2K_Roughness.jpg")[0] == "roughness"
    assert classify_file("readme.txt") == (None, None)


def test_detect_maps(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (8, 8), (200, 120, 80))
    img.save(tmp_path / "Wet_Rock_abc_1K_BaseColor.png")
    Image.new("L", (8, 8), 128).save(tmp_path / "Wet_Rock_abc_1K_Roughness.png")
    Image.new("RGB", (8, 8), (128, 128, 255)).save(tmp_path / "Wet_Rock_abc_1K_Normal.png")
    s = detect_maps(str(tmp_path))
    assert s is not None
    assert s["name"] == "Wet_Rock_abc"
    assert "albedo" in s["maps"]
    assert "roughness" in s["maps"]
