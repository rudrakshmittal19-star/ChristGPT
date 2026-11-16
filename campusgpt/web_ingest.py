"""Fetch public CHRIST pages and dated notices without bypassing robots.txt."""

import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from .config import DATA

WEB = DATA / "web"
EVENTS = WEB / "events"
USER_AGENT = "CampusGPT-student-project/1.0 (public policy and event indexing)"
REQUEST_INTERVAL = 1.0
MAX_EVENT_PAGES = 60
PAGES = [
    "https://christuniversity.in/examination/cia-repeat-regulation",
    "https://christuniversity.in/examination-notifications",
    "https://christuniversity.in/office-of-examinations",
    "https://christuniversity.in/academic-calendar",
    "https://christuniversity.in/admissions",
    "https://christuniversity.in/fee-structure",
    "https://christuniversity.in/grievance-redressal",
    "https://christuniversity.in/library",
    "https://christuniversity.in/placements",
    "https://christuniversity.in/scholarships",
]
EVENT_SEEDS = [
    "https://christuniversity.in/events",
    "https://christuniversity.in/news",
    "https://christuniversity.in/examination-notifications",
    "https://christuniversity.in/admissions",
    "https://christuniversity.in/placements",
]
_last_request = 0.0
_robots = {}


class PageParser(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form"}

    def __init__(self):
        super().__init__()
        self.parts, self.links, self.title, self.headings = [], [], "", []
        self.rows, self._row, self._cell = [], None, None
        self._skip = 0
        self._in_title = False
        self._heading = None
        self._anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("h1", "h2"):
            self._heading = []
        if tag == "a" and attrs.get("href"):
            self._anchor = {"url": attrs["href"], "text": []}
        if tag == "tr":
            self._row = []
        if tag in ("td", "th") and self._row is not None:
            self._cell = []
        if tag == "meta" and attrs.get("name", "").lower() == "description":
            self.parts.append(attrs.get("content", "") + "\n")
        if tag == "time" and attrs.get("datetime"):
            self.parts.append(attrs["datetime"] + " ")
        if tag in ("p", "li", "tr", "div", "h1", "h2", "h3", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("h1", "h2"):
            if self._heading:
                self.headings.append(" ".join(self._heading).strip())
            self._heading = None
        if tag == "a" and self._anchor:
            self.links.append(
                (self._anchor["url"], " ".join(self._anchor["text"]).strip())
            )
            self._anchor = None
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        if tag == "tr" and self._row is not None:
            if self._cell is not None:
                self._row.append(" ".join(self._cell).strip())
                self._cell = None
            if self._row:
                self.rows.append(self._row)
            self._row = None
        if tag in ("p", "li", "tr", "div", "h1", "h2", "h3", "br"):
            self.parts.append("\n")

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text + " "
        elif not self._skip:
            self.parts.append(text + " ")
        if self._heading is not None:
            self._heading.append(text)
        if self._anchor is not None:
            self._anchor["text"].append(text)
        if self._cell is not None:
            self._cell.append(text)


def _request(url):
    global _last_request
    wait = REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if _last_request and wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        if urlparse(response.geturl()).hostname != "christuniversity.in":
            raise PermissionError("refusing a cross-domain redirect")
        return response.read().decode("utf-8", errors="ignore")


def allowed(url):
    parsed = urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname != "christuniversity.in"
    ):
        return False
    path = parsed.path.lower()
    if any(
        part in path for part in ("/login", "/auth", "/dashboard", "/portal", "/erp")
    ):
        return False
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robots:
        try:
            rules = _request(origin + "/robots.txt")
        except (HTTPError, URLError, TimeoutError, OSError):
            return False
        _robots[origin] = rules
    from urllib.robotparser import RobotFileParser

    parser = RobotFileParser()
    parser.set_url(origin + "/robots.txt")
    parser.parse(_robots[origin].splitlines())
    return parser.can_fetch(USER_AGENT, url)


def parse_page(html):
    parser = PageParser()
    parser.feed(html)
    text = re.sub(r"\n\s*\n+", "\n", "".join(parser.parts)).strip()
    title = parser.title.strip()
    if not title or title.lower() == "christ university":
        title = next(iter(parser.headings), "CHRIST University notice")
    return title, text, parser.links, parser.rows


def parse_date(text):
    patterns = (
        r"\b20\d{2}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b",
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2}\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = match.group(0)
        for fmt in (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d %B %Y",
            "%d %b %Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
        ):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
    return ""


def category_for(url, title=""):
    value = (url + " " + title).lower()
    if "exam" in value or "cia" in value or "notification" in value:
        return "exam"
    if "admission" in value:
        return "admissions"
    if "placement" in value or "career" in value:
        return "placements"
    if "news" in value:
        return "news"
    return "events"


def _canonical(url):
    parsed = urlparse(url)
    path = quote(parsed.path.rstrip("/"), safe="/%:@+()_-.")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _candidate(url):
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return False
    return any(
        part in path
        for part in (
            "event",
            "news",
            "notice",
            "notification",
            "admission",
            "placement",
        )
    )


def _without_personal_data(text):
    if re.search(
        r"\b(?:student|candidate|merit|rank)\b.{0,30}\b(?:name|marks|result)\b.{0,20}\blist\b",
        text,
        re.I,
    ):
        return ""
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[email removed]",
        text,
        flags=re.I,
    )
    text = re.sub(r"(?<!\d)(?:\+?91[\s.-]?)?[6-9]\d{9}(?!\d)", "[phone removed]", text)
    text = re.sub(
        r"\b((?:student|roll|registration|application)\s*(?:id|no\.?|number)\s*[:#-]?\s*)[A-Z0-9/-]{4,}\b",
        r"\1[identifier removed]",
        text,
        flags=re.I,
    )
    return text


def fetch(url):
    if not allowed(url):
        raise PermissionError("robots.txt disallows this URL")
    html = _request(url)
    title, text, links, rows = parse_page(html)
    return title, text, links, rows


def _slug(url):
    parsed = urlparse(url)
    value = f"{parsed.path}-{parsed.fragment}"
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "notice"


def collect_events():
    seen = set()
    candidates = []
    for seed in EVENT_SEEDS:
        try:
            title, text, links, table_rows = fetch(seed)
        except Exception as exc:
            print("skipped event seed:", seed, type(exc).__name__)
            continue
        for cells in table_rows:
            dates = [parse_date(cell) for cell in cells]
            date = next((value for value in dates if value), "")
            if not date:
                continue
            title = next(
                (cell for cell in reversed(cells) if cell and not parse_date(cell)), ""
            )
            title = _without_personal_data(title).strip()
            if not title:
                continue
            anchor = _slug(f"{date}-{title}")
            row_url = f"{seed}#notice-{anchor}"
            row_text = _without_personal_data(
                " | ".join(cell for cell in cells if cell)
            ).strip()
            candidates.append((row_url, title, row_text, date))
        for link, label in links:
            target = _canonical(urljoin(seed, link))
            label_lower = label.lower()
            if (
                _candidate(target)
                and urlparse(target).hostname == "christuniversity.in"
                and not any(
                    term in label_lower
                    for term in (
                        "job opening",
                        "research assistant",
                        "advertisement",
                        "walk-in interview",
                    )
                )
            ):
                candidates.append((target, label, "", ""))

    output = []
    for url, title, text, date in candidates[:MAX_EVENT_PAGES]:
        url = _canonical(url)
        if url in seen:
            continue
        seen.add(url)
        if not title or not text:
            try:
                fetched_title, text, _, _ = fetch(url)
                title = title or fetched_title
            except Exception as exc:
                print("skipped event page:", url, type(exc).__name__)
                continue
        text = re.sub(r"\s+", " ", _without_personal_data(text)).strip()
        title = re.sub(r"\s+", " ", _without_personal_data(title)).strip()
        if (
            len(text) < 40
            or not title
            or title.lower() in {"home", "christ university"}
            or title.lower()
            in {
                "main navigation",
                "resources",
                "notices and circulars",
                "empowering minds, inspiring futures",
            }
            or "step into a world of opportunities" in title.lower()
        ):
            continue
        date = date or parse_date(title + " " + text)
        if not date:
            continue
        output.append(
            {
                "title": title[:300],
                "date": date,
                "url": url,
                "category": category_for(url, title),
                "summary": text[:360],
                "text": text[:20000],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return output


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    for url in PAGES:
        try:
            title, text, links, _ = fetch(url)
            title = _without_personal_data(title).strip()
            text = _without_personal_data(text).strip()
            if len(text) < 200:
                print("skipped (too short):", url)
                continue
            (WEB / f"{_slug(url)}.json").write_text(
                json.dumps(dict(url=url, title=title, text=text), ensure_ascii=False),
                encoding="utf-8",
            )
            print("saved public page:", url, len(text), "chars")
        except Exception as exc:
            print("skipped public page:", url, type(exc).__name__)

    EVENTS.mkdir(parents=True, exist_ok=True)
    events = collect_events()
    expected = {f"{_slug(event['url'])}.json" for event in events}
    for stale in EVENTS.glob("*.json"):
        if stale.name not in expected:
            stale.unlink()
    for event in events:
        path = EVENTS / f"{_slug(event['url'])}.json"
        path.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
        print("saved event:", event["title"][:80])


if __name__ == "__main__":
    main()
