#!/usr/bin/env python3
"""
scans/run.py - run your scan categories over the day-trader universe and write
one JSON + one TXT file per category per day, plus docs/data/index.json for the website.

Data comes from the day-trader code you already have (trader-scanner/scan.py): its universe
file, its Yahoo access (query1/query2, 429 back-off) and its New York clock/holiday helpers.
Only the scan RULES (scans/logic.py) are new.

  python3 trader-scanner/scan.py universe                # once, builds trader/universe.txt
  python3 scans/run.py --category eod,stockbee,dt_daily  # after the close
  python3 scans/run.py --category hourly,dt_orb          # during market hours

Steps: 1) one cheap Yahoo call per 20 symbols to drop illiquid names (same as the daily scan);
2) full OHLCV charts for the survivors; 3) every rule in scans/logic.py; 4) write files.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import logic  # noqa: E402

CATEGORIES = {"eod": "End of day", "stockbee": "Stockbee and daily", "hourly": "Hourly",
              "dt_daily": "Day-trader daily setups", "dt_orb": "Day-trader opening range"}
DT_SETUP_META = {  # rule text and direction for the day-trader scanner's own setups
    "breakout_20d": ("Closed above the highest close of the prior 20 days", "bull"),
    "bull_flag": ("Up more than 10% over the prior 10 days, then a 5-day range under 5%", "bull"),
    "reclaim_ma20": ("Closed back above the 20-day average after closing below it", "bull"),
    "oversold": ("RSI under 30", "neutral"),
    "oversold_bounce": ("RSI was under 30 yesterday and the stock closed higher today", "bull"),
}
STOCKBEE_SCREENS = [
    ("sb_momentum", "Stockbee Momentum", "Close at least 2 ATR above the 70-day average, with price above it", "bull", False),
    ("sb_scan1", "Stockbee 1: Extreme Sales Growth", "Revenue up 99%+ year on year", "bull", True),
    ("sb_scan2", "Stockbee 2: Sales Growth", "Revenue up 39%+ in the last quarter and the 2-quarter average", "bull", True),
    ("sb_scan3", "Stockbee 3: Growth / Turnaround", "Scan 2 plus market cap under $11B and IPO within 10 years", "bull", True),
    ("sb_scan4", "Stockbee 4: Earnings + Sales", "EPS up 39%+ in both quarters, revenue up 20%+", "bull", True),
]
FLAGS = [("flag_9mvol", "9M volume + 4% up"), ("flag_plus4_bo", "+4% breakout"),
         ("flag_minus4_bd", "-4% breakdown"), ("flag_ti65", "TI65 bullish")]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_scanner():
    spec = importlib.util.spec_from_file_location("tscan", os.path.join(ROOT, "trader-scanner", "scan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ data (day-trader code)
def to_df(T, ser, intraday):
    cols = {"Open": ser.get("o"), "High": ser.get("h"), "Low": ser.get("l"),
            "Close": ser.get("c"), "Volume": ser.get("v")}
    if any(v is None for v in cols.values()) or not ser.get("t"):
        return None
    idx = pd.DatetimeIndex([T.ts_to_eastern(x) for x in ser["t"]])  # naive New York time
    df = pd.DataFrame(cols, index=idx).apply(pd.to_numeric, errors="coerce")
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if not intraday:
        df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def fetch_df(T, sym, rng, interval, intraday=False, pause=0.0, want_meta=False):
    if pause:
        time.sleep(pause)
    try:
        ser = T.parse_chart(T.yahoo_get(T.chart_path(sym, rng, interval)), sym)
    except Exception:
        return (None, {}) if want_meta else None
    df = to_df(T, ser, intraday)
    return (df, ser.get("meta") or {}) if want_meta else df


def drop_open_bar(T, df, now):
    """Complete daily bars only: drop today's bar until the session has closed."""
    local = T.to_eastern(now)
    today = local.date()
    s = T.session(today)
    done = s is None or local.time() >= s[1]
    cut = pd.Timestamp(today) + (pd.Timedelta(days=1) if done else pd.Timedelta(0))
    return df[df.index < cut]


def monthly_from_daily(d):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    try:
        return d.resample("ME").agg(agg).dropna(subset=["Close"])
    except ValueError:
        return d.resample("M").agg(agg).dropna(subset=["Close"])


def load_symbol(T, sym, need, daily_range, now, complete_daily, pause):
    """All frames the selected categories need for one symbol."""
    d, meta = fetch_df(T, sym, daily_range, "1d", pause=pause, want_meta=True)
    if d is None:
        return None
    if complete_daily:
        d = drop_open_bar(T, d, now)
    if len(d) < 30:
        return None
    out = {"sym": sym, "d": d, "meta": meta}
    if "w" in need:
        out["w"] = fetch_df(T, sym, "5y", "1wk", pause=pause)
    if "m" in need:
        out["m"] = monthly_from_daily(d)
    if "h" in need or "h1" in need:
        h1 = fetch_df(T, sym, "3mo", "1h", intraday=True, pause=pause)
        out["h1"] = h1
        if h1 is not None:
            days = sorted(set(h1.index.normalize()))[-5:]
            out["h"] = h1[h1.index.normalize() >= days[0]]
    if "m15" in need:
        out["m15"] = fetch_df(T, sym, "1mo", "15m", intraday=True, pause=pause)
    if "m5" in need:
        out["m5"] = fetch_df(T, sym, "2d", "5m", intraday=True, pause=pause)
    return out


def stage1(T, symbols, min_price, min_volume, top, pause):
    """Cheap liquidity filter, ranked by dollars traded."""
    data, failed = T.fetch_spark(symbols, "3mo", "1d", pause)
    rows = []
    for s, ser in data.items():
        meta = ser.get("meta") or {}
        p, v = meta.get("regularMarketPrice"), meta.get("regularMarketVolume")
        if p is None or v is None or p < min_price or v < min_volume:
            continue
        rows.append((p * v, s))
    rows.sort(reverse=True)
    syms = [s for _, s in rows]
    return (syms[:top] if top else syms), len(data), data


def _yf():
    import yfinance as yf  # imported late so the other categories work without it
    return yf


def _retry(fn):
    for attempt in (0, 1):
        try:
            return fn()
        except Exception as e:
            if attempt == 0 and "429" in str(e) or "Too Many" in str(e):
                time.sleep(5)
                continue
            return None
    return None


def mcap_yf(sym):
    """Market cap in dollars from Yahoo (yfinance fast_info), or None."""
    v = _retry(lambda: _yf().Ticker(sym).fast_info.market_cap)
    try:
        return float(v) if v and float(v) > 0 else None
    except Exception:
        return None


def fundamentals_yf(sym):
    """Port of sb_get_fundamentals(). Quarters are newest first. Yahoo returns at most 5 quarters."""
    tk = _yf().Ticker(sym)
    out = dict(rev_q=[], eps_q=[], annual_sales_m=None, market_cap=None, currency=None, ipo_years_ago=None)
    out["market_cap"] = mcap_yf(sym)
    fins = _retry(lambda: tk.quarterly_financials)
    if fins is not None and not getattr(fins, "empty", True):
        for label in ["Total Revenue", "TotalRevenue", "Revenue"]:
            if label in fins.index:
                row = fins.loc[label].sort_index(ascending=False)
                out["rev_q"] = [float(v) for v in row.values if not pd.isna(v)]
                break
        if "Diluted EPS" in fins.index:
            row = fins.loc["Diluted EPS"].sort_index(ascending=False)
            out["eps_q"] = [float(v) for v in row.values if not pd.isna(v)]
    qe = _retry(lambda: tk.quarterly_earnings)  # your original source; empty in current yfinance
    if qe is not None and not getattr(qe, "empty", True):
        col = "Earnings" if "Earnings" in qe.columns else qe.columns[0]
        out["eps_q"] = [float(v) for v in qe[col].sort_index(ascending=False).values if not pd.isna(v)]
    if len(out["rev_q"]) >= 4:
        out["annual_sales_m"] = sum(out["rev_q"][:4]) / 1_000_000
    return out


def parallel(fn, items, workers):
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ scanning
def base_row(sym, d, cap=None):
    c = d["Close"]
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) > 1 else last
    row = {"symbol": sym, "price": round(last, 2),
           "chg_pct": round((last / prev - 1) * 100, 2) if prev else 0.0,
           "volume": int(d["Volume"].iloc[-1])}
    if cap:
        row["mktcap_b"], row["cap_src"] = round(cap[0] / 1000, 2), cap[1]
    return row


def run_category(cat, frames, cat_daily, caps, fund_state, relaxed, workers):
    """Registry categories (eod, stockbee, hourly). caps = {sym: (market cap in $M, 'yahoo'|'proxy')}.
    fund_state = None (not run) or dict(fn=callable)."""
    scans = logic.SCANS[cat]
    hits = {k: [] for k, *_ in scans}
    sb = {k: [] for k, *_ in STOCKBEE_SCREENS}
    fund_targets, notes = [], []
    show_cap = cat in ("eod", "stockbee")
    for f in frames:
        d = cat_daily(f)
        cap = caps.get(f["sym"])
        ctx = logic.Ctx(f["sym"], d, f.get("w"), f.get("m"), f.get("h"), f.get("h1"),
                        f.get("m15"), f.get("m5"), cap[0] if cap else 0.0)
        for key, _label, _rule, _dir, fn, need in scans:
            if any(getattr(ctx, n) is None for n in need):
                continue
            try:
                ok = fn(ctx)
            except Exception:
                ok = False
            if ok:
                hits[key].append(base_row(ctx.sym, d, cap if show_cap else None))
        if cat == "stockbee":
            t = logic.stockbee_technical(d.tail(378))
            if t:
                flags = [lab for k, lab in FLAGS if t.get(k)]
                extra = {"ti65": t["ti65"], "ext_atr": t["ext_atr"], "adr_pct": t["adr_pct"], "flags": flags}
                if t["momentum"]:
                    sb["sb_momentum"].append({**base_row(ctx.sym, d, cap), **extra})
                fund_targets.append((ctx.sym, d, t, extra, f.get("meta") or {}, cap))
    if cat == "stockbee" and fund_state and fund_targets:
        log("Fetching fundamentals for %d symbols (slow)..." % len(fund_targets))
        funds = parallel(lambda x: fund_state["fn"](x[0]) or {}, fund_targets, workers)
        n5 = n6 = nonrev = 0
        for (sym, d, t, extra, meta, cap), fund in zip(fund_targets, funds):
            fund = dict(fund)
            if meta.get("currency"):
                fund["currency"] = meta["currency"]
            if meta.get("firstTradeDate"):
                fund["ipo_years_ago"] = (time.time() - float(meta["firstTradeDate"])) / (365.25 * 86400)
            if not fund.get("market_cap") and cap:
                fund["market_cap"] = cap[0] * 1e6
            q = len(fund.get("rev_q") or [])
            n5 += q >= 5; n6 += q >= 6; nonrev += q == 0
            res = logic.stockbee_fundamental(t, fund, relaxed)
            for n in (1, 2, 3, 4):
                if res[f"scan{n}"]:
                    row = {**base_row(sym, d, cap), **extra, "sales_pct": res["sales_pct"],
                           "eps_pct": res["eps_pct"]}
                    if res["relaxed_used"]:
                        row["notes"] = ["relaxed: one quarter stands in for two"]
                    sb[f"sb_scan{n}"].append(row)
        notes.append("Fundamentals read for %d symbols: %d had no revenue data, %d had 5+ quarters, %d had 6+. "
                     "Scans 2-4 need 6 quarters%s." % (len(fund_targets), nonrev, n5, n6,
                     " (relaxed mode is on)" if relaxed else ", so they can only match when that last number is above zero"))
    out = []
    for key, label, rule, direction, *_ in scans:
        rows = sorted(hits[key], key=lambda r: -(r["price"] * r["volume"]))
        out.append({"key": key, "label": label, "rule": rule, "direction": direction,
                    "status": "ok", "count": len(rows), "hits": rows})
    if cat == "stockbee":
        for key, label, rule, direction, needs_fund in STOCKBEE_SCREENS:
            if needs_fund and not fund_state:
                out.append({"key": key, "label": label, "rule": rule, "direction": direction,
                            "status": "not run: fundamentals unavailable", "count": 0, "hits": []})
                continue
            rows = sorted(sb[key], key=lambda r: (-(r.get("sales_pct") or -1e9), -r["ti65"]) if key != "sb_momentum"
                          else (-r["ti65"],))
            out.append({"key": key, "label": label, "rule": rule, "direction": direction,
                        "status": "ok", "count": len(rows), "hits": rows})
    return out, notes


def dt_daily(T, data, cands, now, min_price, min_volume):
    """The day-trader's own daily setups (trader-scanner scan_daily), regrouped per setup."""
    hits, counts, last_date = T.scan_daily(data, cands, now, min_price, min_volume)
    by = {k: [] for k, _ in T.SETUPS}
    for h in hits:
        bars = T.daily_closes(data[h["symbol"]], now)
        prev = bars[-2][1] if len(bars) > 1 else h["price"]
        row = {"symbol": h["symbol"], "price": h["price"],
               "chg_pct": round((h["price"] / prev - 1) * 100, 2) if prev else 0.0,
               "volume": int(h.get("volume") or 0)}
        if h.get("rsi") is not None:
            row["notes"] = ["RSI %.1f" % h["rsi"]]
        for tag in h["setups"]:
            by[tag].append(row)
    scans = []
    for key, label in T.SETUPS:
        rule, direction = DT_SETUP_META.get(key, (label, "neutral"))
        rows = sorted(by[key], key=lambda r: -(r["price"] * r["volume"]))
        scans.append({"key": key, "label": label[0].upper() + label[1:], "rule": rule, "direction": direction,
                      "status": "ok", "count": len(rows), "hits": rows})
    return scans, last_date


def dt_orb(T, cands, now, min_price, pause, or_minutes=15, max_confirm=200):
    """The day-trader's opening-range breakout scan, run at the current New York time."""
    day, cutoff = T.orb_window(now, None, or_minutes, None)
    today = T.to_eastern(T.parse_now(None)).date()
    rng = "1d" if day == today else "5d"
    data, _failed = T.fetch_spark(cands, rng, "5m", pause)

    def confirm(sym):
        if pause:
            time.sleep(pause)
        return T.parse_chart(T.yahoo_get(T.chart_path(sym, rng, "5m")), sym)

    hits, counts, _f = T.scan_orb(data, cands, day, cutoff, or_minutes, min_price, 0, confirm, max_confirm)
    rows = [{"symbol": h["symbol"], "price": h["price"], "chg_pct": None, "volume": None,
             "notes": ["range %s to %s" % (h["or_low"], h["or_high"]), "closed above at %s" % h["at"],
                       "%+.2f%% over" % h["pct_over"]]} for h in hits]
    scan = {"key": "orb", "label": "Opening-range breakout (%d min)" % or_minutes,
            "rule": "First 5-minute bar to close above the high of the first %d minutes" % or_minutes,
            "direction": "bull", "status": "ok", "count": len(rows), "hits": rows}
    note = "Checked %d candidates with real highs and lows%s." % (
        counts["confirmed"], (" (%d more not checked: only the %d most traded are)" % (
            counts["unconfirmed"], max_confirm)) if counts["unconfirmed"] else "")
    return [scan], day, "%s %s ET" % (day.isoformat(), cutoff.strftime("%H:%M")), note


# ------------------------------------------------------------------ files
def write_day(out_dir, date, cat, doc):
    day = os.path.join(out_dir, date)
    os.makedirs(day, exist_ok=True)
    with open(os.path.join(day, cat + ".json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    lines = ["%s | %s | as of %s | generated %s" % (CATEGORIES[cat], date, doc["as_of"], doc["generated_at"]), ""]
    for s in doc["scans"]:
        if s["status"] != "ok":
            lines.append("%s: (%s)" % (s["label"], s["status"]))
        else:
            lines.append("%s (%d): %s" % (s["label"], s["count"], ",".join(h["symbol"] for h in s["hits"])))
    with open(os.path.join(day, cat + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def rebuild_index(out_dir, keep):
    days = sorted((n for n in os.listdir(out_dir)
                   if len(n) == 10 and os.path.isdir(os.path.join(out_dir, n))), reverse=True)
    for old in days[keep:]:
        shutil.rmtree(os.path.join(out_dir, old))
    days = days[:keep]
    index = {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "categories": CATEGORIES, "days": []}
    for day in days:
        entry = {"date": day, "categories": {}}
        for cat in CATEGORIES:
            p = os.path.join(out_dir, day, cat + ".json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    doc = json.load(fh)
                uniq = {h["symbol"] for s in doc["scans"] for h in s["hits"]}
                entry["categories"][cat] = {"file": "%s/%s.json" % (day, cat), "scans": len(doc["scans"]),
                                            "hits": sum(s["count"] for s in doc["scans"]),
                                            "symbols": len(uniq), "generated_at": doc["generated_at"]}
        if entry["categories"]:
            index["days"].append(entry)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, separators=(",", ":"))
    return index


# ------------------------------------------------------------------ main
def main(argv=None, T=None, fund_fn=None, mcap_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="eod,stockbee,dt_daily",
                    help="comma list of: eod, stockbee, hourly, dt_daily, dt_orb")
    ap.add_argument("--universe", default=None, help="symbol file (default trader/universe.txt)")
    ap.add_argument("--symbols", default=None, help="comma list, skips the universe file (testing)")
    ap.add_argument("--min-price", type=float, default=10.0, help="hits under this price are dropped (default 10)")
    ap.add_argument("--min-volume", type=int, default=None, help="default 500000 (intraday-only runs: 300000)")
    ap.add_argument("--top", type=int, default=None, help="keep the N most traded (default: hourly/orb 400, others all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pause", type=float, default=None)
    ap.add_argument("--mktcap", choices=["yahoo", "proxy"], default="yahoo",
                    help="market cap for the rules that use it: Yahoo's figure, or price x 30-day volume (your original proxy)")
    ap.add_argument("--no-fundamentals", action="store_true", help="skip Stockbee scans 1-4 (needs yfinance)")
    ap.add_argument("--sb-relaxed", action="store_true", help="scans 2-4: let one quarter stand in for the missing sixth")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data"))
    ap.add_argument("--keep", type=int, default=31, help="days of history to keep")
    ap.add_argument("--now", default=None, help="test clock, e.g. '2026-10-29 17:00' (New York)")
    a = ap.parse_args(argv)

    cats = [c.strip() for c in a.category.split(",") if c.strip()]
    bad = [c for c in cats if c not in CATEGORIES]
    if bad or not cats:
        ap.error("unknown category %s" % bad)
    T = T or load_scanner()
    pause = T.PAUSE if a.pause is None else a.pause
    now = T.parse_now(a.now)
    reg = [c for c in cats if c in logic.SCANS]
    intraday_only = all(c in ("hourly", "dt_orb") for c in cats)
    min_vol = a.min_volume if a.min_volume is not None else (300_000 if intraday_only else 500_000)
    top = a.top if a.top is not None else (400 if intraday_only else 0)

    if a.symbols:
        symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    else:
        path = a.universe or os.path.join(ROOT, T.DEFAULT_UNIVERSE)
        if not os.path.exists(path):
            log("No %s. Run: python3 trader-scanner/scan.py universe" % path)
            return 2
        symbols = T.symbols_from_file(path)
    log("Universe: %d symbols. Stage 1 (price >= %s, liquidity)..." % (len(symbols), a.min_price))
    cands, answered, spark = stage1(T, symbols, a.min_price, min_vol, top, pause)
    if not cands:
        log("Stage 1 found no symbols (Yahoo answered for %d of %d). Nothing written." % (answered, len(symbols)))
        return 2
    log("%d symbols pass price >= %s and volume >= %s%s." % (len(cands), a.min_price, "{:,}".format(min_vol),
                                                             (" (top %d)" % top) if top else ""))

    frames = []
    if reg:
        need = set(n for c in reg for r in logic.SCANS[c] for n in r[5])
        if "stockbee" in reg:
            need.add("d")
        daily_range = "10y" if "eod" in reg else "2y"
        complete = reg != ["hourly"]
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(load_symbol, T, s, need, daily_range, now, complete, pause) for s in cands]
            for i, fu in enumerate(futs, 1):
                r = fu.result()
                if r and float(r["d"]["Close"].iloc[-1]) >= a.min_price:
                    frames.append(r)
                if i % 100 == 0:
                    log("  loaded %d of %d (%.0fs)" % (i, len(cands), time.time() - t0))
        log("Loaded %d of %d charts." % (len(frames), len(cands)))
        if not frames and not [c for c in cats if c.startswith("dt_")]:
            log("No charts could be loaded. Nothing written.")
            return 2

    # market cap for the rules that use it (high volume, VCP, Satish, Hardik, Stockbee 3)
    caps, cap_note = {}, None
    if frames and any(c in ("eod", "stockbee") for c in reg):
        proxy = {f["sym"]: float(f["d"]["Close"].iloc[-1]) * float(logic.sma(f["d"]["Volume"], 20).iloc[-1]) * 30 / 1e6
                 for f in frames}
        real = {}
        if a.mktcap == "yahoo":
            fn = mcap_fn
            if fn is None:
                try:
                    _yf(); fn = mcap_yf
                except ImportError:
                    log("yfinance is not installed: using the price x volume market cap proxy.")
            if fn:
                log("Reading market caps for %d symbols..." % len(frames))
                for f, v in zip(frames, parallel(lambda f: fn(f["sym"]), frames, a.workers)):
                    if v:
                        real[f["sym"]] = v / 1e6
        for sym, p in proxy.items():
            caps[sym] = (real[sym], "yahoo") if sym in real else (p, "proxy")
        cap_note = "Market cap: Yahoo's figure for %d of %d symbols, price x 30-day volume for the rest." % (len(real), len(frames))

    fund_state = None
    if "stockbee" in reg and not a.no_fundamentals:
        fn = fund_fn
        if fn is None:
            try:
                _yf(); fn = fundamentals_yf
            except ImportError:
                log("yfinance is not installed: Stockbee scans 1-4 will not run.")
        if fn:
            fund_state = {"fn": fn}

    gen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    local = T.to_eastern(now)
    wrote = 0
    for cat in cats:
        notes = ["A scan is a list of charts that match a rule, not a list of stocks to buy."]
        try:
            if cat in logic.SCANS:
                cat_daily = (lambda f: f["d"]) if cat == "eod" else (lambda f: f["d"].tail(504))
                scans, extra = run_category(cat, frames, cat_daily, caps, fund_state, a.sb_relaxed, a.workers)
                notes += extra
                if cat in ("eod", "stockbee") and cap_note:
                    notes.append(cap_note)
                if cat == "hourly":
                    date = local.date().isoformat()
                    as_of = "%s %s ET (daily bar still forming)" % (date, local.strftime("%H:%M"))
                else:
                    last = pd.Series([f["d"].index[-1] for f in frames]).mode().iloc[0]
                    date, as_of = last.date().isoformat(), "%s close" % last.date().isoformat()
                loaded = len(frames)
            elif cat == "dt_daily":
                scans, last = dt_daily(T, spark, cands, now, a.min_price, min_vol)
                if not last:
                    log("dt_daily: no daily data. Skipped.")
                    continue
                date, as_of, loaded = last.isoformat(), "%s close" % last.isoformat(), len(cands)
            else:
                scans, day, as_of, note = dt_orb(T, cands, now, a.min_price, pause)
                notes.append(note)
                date, loaded = day.isoformat(), len(cands)
        except T.ScanError as e:
            log("%s: skipped. %s" % (cat, e))
            continue
        doc = {"category": cat, "label": CATEGORIES[cat], "date": date, "as_of": as_of, "generated_at": gen,
               "universe": len(symbols), "candidates": len(cands), "loaded": loaded,
               "filters": {"min_price": a.min_price, "min_volume": min_vol, "top": top or None},
               "notes": notes, "scans": scans}
        os.makedirs(a.out, exist_ok=True)
        write_day(a.out, date, cat, doc)
        wrote += 1
        log("%s: %s -> %d hits in %d scans" % (cat, date, sum(s["count"] for s in scans), len(scans)))
    if not wrote:
        log("Nothing was written.")
        return 2
    rebuild_index(a.out, a.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
