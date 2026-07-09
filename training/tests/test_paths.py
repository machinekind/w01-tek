from fbb_rl import paths


def test_source_model_exists():
    assert paths.SOURCE_XML.exists()
    assert paths.SOURCE_XML.name == "four_bar_bot.xml"


def test_legs():
    assert len(paths.LEGS) == 4
