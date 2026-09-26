from datetime import datetime, timezone
from core.news_checker import NewsFilter


def test_news_filter_high_impact(tmp_path):
    news_file = tmp_path / "events.json"
    news_file.write_text('[{"timestamp_utc":"2026-02-07T14:30:00+00:00","currency":"USD","title":"NFP","impact":"HIGH"}]')
    nf = NewsFilter(str(news_file))
    ts = datetime(2026, 2, 7, 14, 15, tzinfo=timezone.utc)
    assert nf.high_impact_at(ts) is True
