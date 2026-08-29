from wojtek_rl import paths


def test_source_model_exists():
    assert paths.SOURCE_XML.exists()
    assert paths.SOURCE_XML.name == "wojtek.xml"


def test_legs():
    assert len(paths.LEGS) == 4
