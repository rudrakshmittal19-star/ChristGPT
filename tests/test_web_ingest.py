from campusgpt.web_ingest import (
    _without_personal_data,
    category_for,
    parse_date,
    parse_page,
    _slug,
)


def test_parse_public_notice_metadata():
    title, text, links, rows = parse_page(
        "<html><title>CHRIST University</title><h1>Campus Open Day</h1>"
        "<table><tr><th>Date</th><th>Title</th></tr><tr><td>01/12/2026</td><td>Open Day</td></tr></table>"
        '<a href="/events/open-day">Details</a></html>'
    )
    assert title == "Campus Open Day"
    assert "Campus Open Day" in text
    assert links == [("/events/open-day", "Details")]
    assert rows[-1] == ["01/12/2026", "Open Day"]
    assert parse_date("01/12/2026") == "2026-12-01"
    assert _slug("https://christuniversity.in/notices#notice-1") != _slug(
        "https://christuniversity.in/notices#notice-2"
    )
    assert (
        category_for("https://christuniversity.in/examination-notifications") == "exam"
    )


def test_scraper_redacts_contact_and_student_identifiers():
    cleaned = _without_personal_data(
        "Contact person@example.edu or +91 9876543210. Student ID: 21ABC12345"
    )
    assert "person@example.edu" not in cleaned
    assert "9876543210" not in cleaned
    assert "21ABC12345" not in cleaned
    assert "[email removed]" in cleaned
    assert "[phone removed]" in cleaned
    assert "[identifier removed]" in cleaned
    assert _without_personal_data("Student name marks list") == ""
