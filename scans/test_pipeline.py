"""Offline self-test: fake Yahoo answers, no network. Run:  python3 scans/test_pipeline.py"""
import datetime as dt, json, os, re, sys, tempfile, urllib.parse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run, logic

END = dt.date(2026, 10, 29)  # a Thursday


def bdays(n, end=END):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5: out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


def ts(d, hh=14, mm=30):
    return int(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.timezone.utc).timestamp())


def walk(n, seed, drift, start=50.0, vol0=2_000_000):
    r = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(r.normal(drift, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + abs(r.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - abs(r.normal(0, 0.006, n)))
    v = vol0 * (1 + abs(r.normal(0, 0.3, n)))
    return o, h, l, c, v


def chart(timestamps, o, h, l, c, v, price=None, vol=None):
    meta = {"regularMarketPrice": price if price is not None else float(c[-1]),
            "regularMarketVolume": int(vol if vol is not None else v[-1])}
    q = {"open": list(map(float, o)), "high": list(map(float, h)), "low": list(map(float, l)),
         "close": list(map(float, c)), "volume": list(map(int, v))}
    return {"timestamp": timestamps, "indicators": {"quote": [q]}, "meta": meta}


SYMS = {"AAA": 0.0012, "BBB": -0.0008, "CCC": 0.0004, "PENNY": 0.0, "CHEAP": 0.0}


def make(sym, rng, interval):
    seed = sum(map(ord, sym)); drift = SYMS[sym]; start = {"PENNY": 2.0, "CHEAP": 5.2}.get(sym, 50.0)
    if interval == "1d":
        n = {"5d": 5, "3mo": 63, "2y": 504, "10y": 2500}[rng]; days = bdays(n)
        o, h, l, c, v = walk(2500, seed, drift, start)
        o, h, l, c, v = (x[-n:] for x in (o, h, l, c, v))
        if sym == "AAA":  # make the last bar a big green volume bar
            c[-1] = c[-2] * 1.05; o[-1] = c[-2] * 1.0; h[-1] = c[-1] * 1.003; l[-1] = o[-1] * 0.998; v[-1] = v.max() * 1.2
        return chart([ts(d) for d in days], o, h, l, c, v, vol=v[-1] if sym != "PENNY" else 50_000)
    if interval == "1wk":
        days = [d for d in bdays(5 * 52 * 5) if d.weekday() == 0][-260:]
        o, h, l, c, v = walk(len(days), seed + 1, drift * 5, start, 10_000_000)
        return chart([ts(d) for d in days], o, h, l, c, v)
    step = {"1h": 60, "15m": 15, "5m": 5}[interval]
    nd = {"3mo": 63, "1mo": 21, "5d": 5, "2d": 2, "1d": 1}[rng]; stamps = []
    for d in bdays(nd):
        t = 13 * 60 + 30
        while t < 20 * 60: stamps.append(ts(d, t // 60, t % 60)); t += step
    o, h, l, c, v = walk(len(stamps), seed + 2, drift / 20, start, 100_000)
    if interval == "5m" and sym == "AAA":  # flat opening range, then a clean break above it
        k = np.arange(len(c)); base = 100.0
        c = np.where(k < 3, base + 0.05 * (k % 2), base + 0.8 + 0.05 * k)
        o = np.r_[base, c[:-1]]; h = np.maximum(o, c) + 0.05; l = np.minimum(o, c) - 0.05
    return chart(stamps, o, h, l, c, v)


def fake_yahoo(path):
    q = urllib.parse.urlparse(path); qs = urllib.parse.parse_qs(q.query)
    rng, interval = qs["range"][0], qs["interval"][0]
    if "/spark" in q.path:
        res = []
        for s in qs["symbols"][0].split(","):
            if s in SYMS:
                res.append({"symbol": s, "response": [make(s, rng, interval)]})
        return json.dumps({"spark": {"result": res, "error": None}})
    sym = q.path.rsplit("/", 1)[1]
    if sym not in SYMS: return json.dumps({"chart": {"result": None, "error": {"description": "No data found"}}})
    return json.dumps({"chart": {"result": [make(sym, rng, interval)], "error": None}})


def fake_fund(sym):
    if sym != "AAA": return {"rev_q": [], "eps_q": []}
    rev = [300e6, 280e6, 250e6, 220e6, 100e6]            # five quarters, like Yahoo; +200% year on year
    return {"rev_q": rev, "eps_q": [1.0, 0.8, 0.6, 0.5, 0.2], "market_cap": 3e9, "annual_sales_m": sum(rev[:4]) / 1e6}


def main():
    T = run.load_scanner(); T.yahoo_get = fake_yahoo; T.SLEEP = lambda s: None
    out = tempfile.mkdtemp()
    base = ["--symbols", ",".join(list(SYMS) + ["NOPE"]), "--pause", "0", "--out", out, "--workers", "2"]
    kw = dict(fund_fn=fake_fund, mcap_fn=lambda s: 7e9)
    assert run.main(["--category", "eod,stockbee,dt_daily", "--now", "2026-10-29 17:00"] + base, T, **kw) == 0
    assert run.main(["--category", "hourly,dt_orb", "--now", "2026-10-29 10:35"] + base, T, **kw) == 0
    idx = json.load(open(os.path.join(out, "index.json")))
    assert [d["date"] for d in idx["days"]] == ["2026-10-29"], idx
    cats = idx["days"][0]["categories"]; assert set(cats) == {"eod", "stockbee", "hourly", "dt_daily", "dt_orb"}, cats
    for cat in cats:
        doc = json.load(open(os.path.join(out, "2026-10-29", cat + ".json")))
        keys = [s["key"] for s in doc["scans"]]
        assert len(keys) == len(set(keys))
        syms = {h["symbol"] for s in doc["scans"] for h in s["hits"]}
        assert not syms & {"PENNY", "CHEAP"}, (cat, syms)                  # the $10 price floor holds everywhere
        assert all(h["price"] >= 10 for s in doc["scans"] for h in s["hits"])
        print("%-9s scans=%2d hits=%3d symbols=%s  %s" % (cat, len(keys), cats[cat]["hits"], sorted(syms), [s["key"] for s in doc["scans"] if s["count"]][:5]))
    sb = json.load(open(os.path.join(out, "2026-10-29", "stockbee.json")))
    s1 = [s for s in sb["scans"] if s["key"] == "sb_scan1"][0]
    assert s1["status"] == "ok" and [h["symbol"] for h in s1["hits"]] == ["AAA"], s1          # scan 1 works with 5 quarters
    for k in ("sb_scan2", "sb_scan3", "sb_scan4"):
        assert [s for s in sb["scans"] if s["key"] == k][0]["count"] == 0                      # fail-closed without a 6th quarter
    assert any("6+" in n or "6+ " in n for n in sb["notes"]) and any("Market cap" in n for n in sb["notes"]), sb["notes"]
    eod = json.load(open(os.path.join(out, "2026-10-29", "eod.json")))
    row = next(h for s in eod["scans"] for h in s["hits"]); assert row["cap_src"] == "yahoo" and row["mktcap_b"] == 7.0, row
    orb = json.load(open(os.path.join(out, "2026-10-29", "dt_orb.json")))
    assert [h["symbol"] for h in orb["scans"][0]["hits"]] == ["AAA"] and orb["scans"][0]["hits"][0]["chg_pct"] is None, orb["scans"]
    # relaxed mode lets scans 2-4 see one quarter in place of the sixth
    out2 = tempfile.mkdtemp()
    assert run.main(["--category", "stockbee", "--now", "2026-10-29 17:00", "--sb-relaxed"] + base[:-2] + ["--out", out2], T, **kw) == 0
    sb2 = json.load(open(os.path.join(out2, "2026-10-29", "stockbee.json")))
    assert [s for s in sb2["scans"] if s["key"] == "sb_scan2"][0]["count"] == 1
    # market closed: ORB is skipped without writing, other categories still written
    out3 = tempfile.mkdtemp()
    assert run.main(["--category", "dt_orb", "--now", "2026-10-31 11:00"] + base[:-2] + ["--out", out3], T, **kw) == 2
    # pruning
    for i in range(1, 40): os.makedirs(os.path.join(out, "2026-09-%02d" % (i % 28 + 1) if i < 28 else "2026-08-%02d" % (i - 27)), exist_ok=True)
    run.rebuild_index(out, 31)
    assert len([n for n in os.listdir(out) if os.path.isdir(os.path.join(out, n))]) <= 31
    import pandas as pd
    mk = lambda rows: pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    up = mk([[10, 11, 9.8, 10.6, 1], [10.5, 10.8, 9.9, 10.0, 1], [9.9, 11.0, 9.5, 10.9, 1], [10.8, 11.0, 9.4, 10.95, 1]])
    assert logic.scan_trend_reversal(up.iloc[:3].reset_index(drop=True).pipe(lambda x: pd.concat([x.iloc[:1], x], ignore_index=True))) is True
    assert logic.scan_trend_reversal(up.iloc[:2].reset_index(drop=True).pipe(lambda x: pd.concat([x, x], ignore_index=True))) is False
    wk = mk([[10, 11, 9, 9.5, 1], [9.4, 9.8, 9.2, 9.3, 1], [9.2, 10.5, 9.0, 10.4, 1]])
    assert logic.scan_weekly_reversal_engulfing(None, wk) is True
    assert logic.scan_weekly_reversal_engulfing(None, wk.iloc[:2]) is False
    print("all checks passed")


main()
