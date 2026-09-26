import argparse
import csv
import json
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable, Dict, Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

COUNTRY_TO_CCY = {
    "United States": "USD",
    "Euro Area": "EUR",
    "Japan": "JPY",
    "United Kingdom": "GBP",
    "Switzerland": "CHF",
    "Australia": "AUD",
    "Canada": "CAD",
    "China": "CNY",
}

import phoenix_config as config


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
    except Exception:
        return None


def _normalize_event(raw: Dict[str, Any]) -> Dict[str, str] | None:
    ts = raw.get("timestamp_utc") or raw.get("timestamp") or raw.get("time") or raw.get("Date")
    currency = raw.get("currency") or raw.get("ccy") or raw.get("Currency") or ""
    if not currency:
        country = raw.get("Country") or ""
        currency = COUNTRY_TO_CCY.get(country, "")
    title = raw.get("title") or raw.get("event") or raw.get("name") or raw.get("Event") or ""
    impact = raw.get("impact") or raw.get("importance") or raw.get("Importance") or ""
    if not ts:
        return None
    parsed = _parse_ts(str(ts))
    if not parsed:
        return None
    impact_value = str(impact).upper()
    if isinstance(impact, (int, float)):
        impact_value = "HIGH" if impact >= 3 else "MEDIUM" if impact == 2 else "LOW"
    return {
        "timestamp_utc": parsed.isoformat(),
        "currency": str(currency),
        "title": str(title),
        "impact": impact_value,
    }


def _filter_years(events: Iterable[Dict[str, str]], start_year: int, end_year: int) -> list[Dict[str, str]]:
    filtered = []
    for ev in events:
        parsed = _parse_ts(ev["timestamp_utc"])
        if not parsed:
            continue
        if start_year <= parsed.year <= end_year:
            filtered.append(ev)
    return filtered


def _load_from_csv(text: str) -> list[Dict[str, str]]:
    reader = csv.DictReader(StringIO(text))
    events = []
    for row in reader:
        ev = _normalize_event(row)
        if ev:
            events.append(ev)
    return events


def _load_from_json(text: str) -> list[Dict[str, str]]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("events", [])
    events = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ev = _normalize_event(item)
        if ev:
            events.append(ev)
    return events


def _extract_events_from_investing_html(html: str) -> list[Dict[str, str]]:
    events = []
    rows = re.findall(r"<tr[^>]*class=\"[^\"]*js-event-item[^\"]*\"[^>]*>.*?</tr>", html, flags=re.S)
    for row in rows:
        dt_match = re.search(r"data-event-datetime=\"([^\"]+)\"", row)
        if not dt_match:
            continue
        dt_str = dt_match.group(1)
        try:
            dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
        except Exception:
            continue

        cur_match = re.search(r"flagCur[^>]*>.*?</span>\s*([A-Z]{3})", row, flags=re.S)
        if not cur_match:
            continue
        currency = cur_match.group(1).strip()

        impact_match = re.search(r"data-img_key=\"bull(\d)\"", row)
        impact = ""
        if impact_match:
            lvl = int(impact_match.group(1))
            impact = "HIGH" if lvl >= 3 else "MEDIUM" if lvl == 2 else "LOW"

        title_match = re.search(r"class=\"left event[^\"]*\"[^>]*>(.*?)</td>", row, flags=re.S)
        title = ""
        if title_match:
            raw_title = title_match.group(1)
            title = re.sub(r"<[^>]+>", "", raw_title).strip()

        events.append(
            {
                "timestamp_utc": dt.isoformat(),
                "currency": currency,
                "title": title,
                "impact": impact,
            }
        )
    return events


def download_investing(
    out_path: Path,
    start_date: datetime,
    end_date: datetime,
    chunk_days: int = 30,
    cooldown_seconds: int = 0,
) -> int:
    events: list[Dict[str, str]] = []
    url = "https://es.investing.com/economic-calendar/Service/getCalendarFilteredData"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://es.investing.com/economic-calendar/",
    }
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        params = {
            "dateFrom": cursor.strftime("%Y-%m-%d"),
            "dateTo": chunk_end.strftime("%Y-%m-%d"),
            "timeFilter": "timeRemain",
            "currentTab": "custom",
            "limit_from": 0,
        }
        req = Request(url, data=urlencode(params).encode("utf-8"), headers=headers)
        for attempt in range(5):
            try:
                with urlopen(req) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if exc.code == 429 and attempt < 4:
                    time.sleep(5 + attempt * 5)
                    continue
                raise
        html = payload.get("data", "")
        events.extend(_extract_events_from_investing_html(html))
        time.sleep(2.0)
        cursor = chunk_end + timedelta(days=1)

    currencies = set(getattr(config, "NEWS_CURRENCIES", []))
    if currencies:
        events = [ev for ev in events if ev.get("currency") in currencies]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "currency", "title", "impact"])
        writer.writeheader()
        writer.writerows(events)

    return len(events)


def download_news(url: str, out_path: Path, start_year: int, end_year: int) -> int:
    with urlopen(url) as response:
        raw = response.read().decode("utf-8")

    if url.lower().endswith(".json"):
        events = _load_from_json(raw)
    else:
        events = _load_from_csv(raw)

    events = _filter_years(events, start_year, end_year)
    currencies = set(getattr(config, "NEWS_CURRENCIES", []))
    if currencies:
        events = [ev for ev in events if ev.get("currency") in currencies]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "currency", "title", "impact"])
        writer.writeheader()
        writer.writerows(events)

    return len(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize news events.")
    parser.add_argument("--url", help="CSV or JSON URL with news events")
    parser.add_argument("--source", choices=["url", "investing"], default="url")
    parser.add_argument("--out", default=config.NEWS_FEED_FILE, help="Output CSV path")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--cooldown-seconds", type=int, default=0)
    args = parser.parse_args()

    if args.source == "investing":
        if not args.start_date or not args.end_date:
            raise SystemExit("start-date y end-date son obligatorios para source=investing")
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
        count = download_investing(
            Path(args.out),
            start_date,
            end_date,
            chunk_days=args.chunk_days,
            cooldown_seconds=args.cooldown_seconds,
        )
    else:
        if not args.url:
            raise SystemExit("URL requerida para source=url")
        count = download_news(args.url, Path(args.out), args.start_year, args.end_year)
    print(f"✅ Downloaded {count} events to {args.out}")


if __name__ == "__main__":
    main()
