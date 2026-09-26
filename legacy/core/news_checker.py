import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
from typing import List, Optional

import phoenix_config as config


@dataclass
class NewsEvent:
    timestamp_utc: datetime
    currency: str
    title: str
    impact: str


class NewsFilter:
    """Carga eventos de noticias y decide si pausar trading."""

    def __init__(self, news_file: Optional[str] = None):
        self.news_file = Path(news_file or config.NEWS_EVENTS_FILE)
        self.feed_file = Path(config.NEWS_FEED_FILE)

    def _load_from_csv_feed(self) -> List[NewsEvent]:
        if not self.feed_file.exists():
            return []
        events: List[NewsEvent] = []
        with self.feed_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row.get("timestamp_utc", "")).replace(tzinfo=timezone.utc)
                    events.append(
                        NewsEvent(
                            timestamp_utc=ts,
                            currency=row.get("currency", "") or "",
                            title=row.get("title", "") or "",
                            impact=row.get("impact", "") or "",
                        )
                    )
                except Exception:
                    continue
        return events

    def _load_events(self) -> List[NewsEvent]:
        feed_events = self._load_from_csv_feed()
        if feed_events:
            return feed_events
        if not self.news_file.exists():
            return []
        data = json.loads(self.news_file.read_text())
        events = []
        for item in data:
            try:
                ts = datetime.fromisoformat(item["timestamp_utc"]).replace(tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        timestamp_utc=ts,
                        currency=item.get("currency", "") or "",
                        title=item.get("title", "") or "",
                        impact=item.get("impact", "") or "",
                    )
                )
            except Exception:
                continue
        return events

    def high_impact_soon(self) -> bool:
        now = datetime.now(timezone.utc)
        return self.high_impact_at(now)

    def high_impact_at(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        events = self._load_events()
        if not events:
            return False
        window_before = timedelta(minutes=config.STOP_BEFORE_HIGH_IMPACT_NEWS_MIN)
        window_after = timedelta(minutes=config.RESUME_AFTER_NEWS_MIN)
        for ev in events:
            if ev.impact.upper() != "HIGH":
                continue
            if ev.currency and ev.currency not in config.NEWS_CURRENCIES:
                continue
            if ts - window_after <= ev.timestamp_utc <= ts + window_before:
                return True
        return False
