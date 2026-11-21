from fuse.tools.basic import make_basic_tools


def test_current_time_in_a_timezone():
    (get_current_time,) = make_basic_tools("UTC")
    assert "UTC" in get_current_time.invoke({"timezone": "UTC"})


def test_unknown_timezone_is_reported_not_raised():
    (get_current_time,) = make_basic_tools("UTC")
    assert get_current_time.invoke({"timezone": "Mars/Base"}) == "Unknown timezone: Mars/Base"
