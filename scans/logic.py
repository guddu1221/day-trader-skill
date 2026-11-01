"""
scans/logic.py - scan RULES only.

Extracted from your three notebooks:
  scanner_volume_eod_v5.py          -> category "eod"
  scanner_daily_stockbee_updated.py -> category "stockbee"
  scanner_hourly_v2.py              -> category "hourly"

Everything that was Colab, Google Sheets, Supabase, ticker-list scraping or yfinance
downloading has been dropped. Each rule takes pandas DataFrames with the columns
Open, High, Low, Close, Volume and returns True/False. Rule bodies are kept as in
your files; the few places where I changed behaviour are marked  # CHANGED.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------- indicators
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi_sma(s, n=14):
    """Your daily/hourly files' rsi(): simple rolling mean."""
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def rsi_wilder(s, n=14):
    """Your new-EOD/BTST _rsi(): Wilder smoothing."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))


def _f(x):
    try: return float(x)
    except Exception: return float("nan")


def _range(d, i):
    try: return _f(d["High"].iloc[-(i + 1)]) - _f(d["Low"].iloc[-(i + 1)])
    except Exception: return float("nan")


def vwap_intraday(h):
    day = h.index.normalize()
    tp = (h["High"] + h["Low"] + h["Close"]) / 3
    return (tp * h["Volume"]).groupby(day).cumsum() / h["Volume"].groupby(day).cumsum()


# ================================================================= EOD scans
def scan_hv(d, w, mktcap_m):
    """High Volume: today's volume is a 63d / 252d / 2000d high, plus quality filters."""
    try:
        if len(d) < 63 or len(w) < 12: return False
        vol = float(d["Volume"].iloc[-1]); close = float(d["Close"].iloc[-1])
        prev_close = float(d["Close"].iloc[-2])
        if vol <= 0 or close <= 0: return False
        lookback = min(len(d), 2000)
        max_2000 = d["Volume"].rolling(lookback, min_periods=lookback).max().iloc[-1]
        max_252 = d["Volume"].rolling(min(252, len(d)), min_periods=252).max().iloc[-1]
        max_63 = d["Volume"].rolling(63, min_periods=63).max().iloc[-1]
        group_a = ((pd.notna(max_2000) and vol >= float(max_2000)) or
                   (pd.notna(max_252) and vol >= float(max_252)) or
                   (pd.notna(max_63) and vol >= float(max_63)))
        if not group_a: return False
        avg_wk_vol = sma(w["Volume"], 10).iloc[-1]
        avg_vol_50 = sma(d["Volume"], 50).iloc[-1]
        if pd.isna(avg_wk_vol) or pd.isna(avg_vol_50): return False
        return (close >= 20 and mktcap_m >= 100 and float(avg_wk_vol) > 100000
                and float(avg_vol_50) * close > 5000000 and close >= prev_close)
    except Exception:
        return False


def scan_weekly_rsi_reversal(d, w):
    try:
        if w is None or len(w) < 15 or len(d) < 15: return False
        if _f(rsi_wilder(w["Close"], 14).iloc[-1]) <= 50: return False
        sma10 = _f(d["Close"].rolling(10).mean().iloc[-1])
        if not (_f(d["Close"].iloc[-1]) > sma10): return False
        if not (_f(d["Close"].rolling(10).mean().iloc[-6]) > _f(d["Close"].iloc[-6])): return False
        if not (_f(d["Close"].iloc[-2]) < _f(d["Open"].iloc[-2])): return False
        return _f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-1])
    except Exception:
        return False


def scan_btst_strong(d):
    try:
        if len(d) < 15: return False
        ema100 = _f(d["Close"].ewm(span=100, adjust=False).mean().iloc[-1])
        vol = _f(d["Volume"].iloc[-1]); close = _f(d["Close"].iloc[-1])
        prev_h = _f(d["High"].iloc[-2]); v_sma10 = _f(d["Volume"].rolling(10).mean().iloc[-1])
        rsi_val = _f(rsi_wilder(d["Close"], 14).iloc[-1])
        return (close > ema100 and vol > 500_000 and close > 100 and rsi_val > 55
                and close > prev_h and vol > v_sma10 * 2)
    except Exception:
        return False


def scan_btst_supertrend(d):
    """Supertrend here is the ATR lower-band approximation from your file, not a true supertrend."""
    try:
        if len(d) < 60: return False
        close, high, low = d["Close"], d["High"], d["Low"]
        atr = (high - low).rolling(7).mean() * 3
        st = ((high + low) / 2 - atr).iloc[-1]
        c0 = _f(close.iloc[-1]); c1 = _f(close.iloc[-2]); c2 = _f(close.iloc[-3])
        o0 = _f(d["Open"].iloc[-1])
        wk52_high = _f(high.rolling(min(252, len(high))).max().iloc[-1])
        if not (c0 > o0 and c0 > c1 and c1 >= c2): return False
        if not (c0 > st): return False
        if not (_f(high.iloc[-1]) >= wk52_high * 0.99): return False
        return _f(rsi_wilder(close, 14).iloc[-1]) > 50
    except Exception:
        return False


def scan_trend_reversal(d):
    try:
        if len(d) < 4: return False
        return (_f(d["High"].iloc[-1]) > _f(d["High"].iloc[-2]) and
                _f(d["Low"].iloc[-1]) < _f(d["Low"].iloc[-2]) and
                _f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-1]) and
                _f(d["Close"].iloc[-2]) < _f(d["Open"].iloc[-2]))
    except Exception:
        return False


def scan_weekly_reversal_engulfing(d, w):
    try:
        if w is None or len(w) < 3: return False
        return (_f(w["High"].iloc[-1]) > _f(w["High"].iloc[-2]) and
                _f(w["Low"].iloc[-1]) < _f(w["Low"].iloc[-2]) and
                _f(w["Close"].iloc[-1]) > _f(w["Open"].iloc[-1]) and
                _f(w["Close"].iloc[-2]) < _f(w["Open"].iloc[-2]))
    except Exception:
        return False


# ============================================================ daily scans (stockbee file)
def scan_bearish_momentum(d):
    try:
        if len(d) < 52: return False
        c = _f(d["Close"].iloc[-1])
        return (c > 50 and _f(rsi_sma(d["Close"]).iloc[-1]) < 50
                and c < _f(d["Low"].iloc[-2]) and c < _f(sma(d["Close"], 50).iloc[-1])
                and _f(sma(d["Volume"], 20).iloc[-1]) > 500000)
    except Exception:
        return False


def scan_vcp(d, mktcap_m):
    try:
        if len(d) < 252: return False
        c = _f(d["Close"].iloc[-1]); dvol = c * _f(sma(d["Volume"], 20).iloc[-1])
        s200 = _f(sma(d["Close"], 200).iloc[-1]); s50 = _f(sma(d["Close"], 50).iloc[-1])
        c22 = _f(d["Close"].iloc[-23]); c66 = _f(d["Close"].iloc[-67])
        hi = _f(d["High"].rolling(252).max().iloc[-1])
        return ((c / c22 > 1.2 and mktcap_m > 1 and dvol > 30e6 and c > s200) or
                (c / c66 >= 1.3 and c >= 1 and dvol > 30e6 and c > s200) or
                (mktcap_m >= 1000 and c > hi * 0.75 and c > s50 and c > s200 and dvol > 30e6))
    except Exception:
        return False


def scan_satish_bullish(d, mktcap_m):
    try:
        if len(d) < 47: return False
        c = _f(d["Close"].iloc[-1]); o = _f(d["Open"].iloc[-1])
        lo = _f(d["Low"].iloc[-1]); v = _f(d["Volume"].iloc[-1])
        pc = _f(d["Close"].iloc[-2]); pl = _f(d["Low"].iloc[-2]); ph = _f(d["High"].iloc[-2])
        pivot = (ph + pl + pc) / 3; r = rsi_sma(d["Close"])
        return (mktcap_m >= 1000 and c >= 100 and c > pivot
                and v > _f(sma(d["Volume"], 10).iloc[-1])
                and lo > pl and c > pc and c > o
                and _f(sma(d["Close"], 9).iloc[-1]) > _f(ema(d["Close"], 45).iloc[-1])
                and _f(r.iloc[-3]) > _f(r.iloc[-2])
                and (c - pc) / pc * 100 > 1 and _f(r.iloc[-1]) > 30 and v > 100000)
    except Exception:
        return False


def scan_hardik(d, w, mktcap_m):
    try:
        if len(d) < 12 or w is None or len(w) < 16: return False
        return (_f(rsi_sma(w["Close"]).iloc[-1]) > 50
                and _f(d["Close"].iloc[-1]) > _f(sma(d["Close"], 10).iloc[-1])
                and _f(sma(d["Close"], 10).iloc[-6]) > _f(d["Close"].iloc[-6])
                and _f(d["Close"].iloc[-2]) < _f(d["Open"].iloc[-2])
                and _f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-1])
                and mktcap_m > 5000)
    except Exception:
        return False


def scan_nr6(d):
    try:
        if len(d) < 8: return False
        r0 = _range(d, 0)
        return all(r0 < _range(d, i) for i in range(1, 7))
    except Exception:
        return False


def scan_rocket_base(d):
    try:
        if len(d) < 92: return False
        c = _f(d["Close"].iloc[-1])
        if c <= 30 or _f(sma(d["Volume"], 50).iloc[-1]) < 50000: return False
        return (c >= _f(d["Low"].iloc[-6]) * 1.2 or c >= _f(d["Low"].iloc[-31]) * 1.3
                or c >= _f(d["Low"].iloc[-91]) * 1.3)
    except Exception:
        return False


def scan_ep_breakout(d):
    """Same rule as the hourly file's Short Term Breakout, but needs only 130 bars."""
    try:
        if len(d) < 130: return False
        max5 = _f(d["Close"].rolling(5).max().iloc[-1])
        max120_6ago = _f(d["Close"].rolling(120).max().iloc[-7])
        return (max5 > max120_6ago * 1.05
                and _f(d["Volume"].iloc[-1]) > _f(sma(d["Volume"], 5).iloc[-1])
                and _f(d["Close"].iloc[-1]) > _f(d["Close"].iloc[-2]))
    except Exception:
        return False


def scan_20day_high(d):
    try:
        if len(d) < 21: return False
        return _f(d["Close"].iloc[-1]) >= _f(d["High"].rolling(20).max().iloc[-1])
    except Exception:
        return False


def scan_2month_high(d):
    try:
        if len(d) < 43: return False
        return _f(d["Close"].iloc[-1]) >= _f(d["High"].rolling(42).max().iloc[-1])
    except Exception:
        return False


def scan_gapup(d):
    try:
        if len(d) < 2: return False
        return _f(d["Open"].iloc[-1]) >= _f(d["Close"].iloc[-2]) * 1.03
    except Exception:
        return False


def scan_up4pct(d):
    try:
        if len(d) < 2: return False
        c = _f(d["Close"].iloc[-1]); o = _f(d["Open"].iloc[-1])
        return (c - o) / o >= 0.04
    except Exception:
        return False


# ============================================================ Stockbee screener (stockbee file)
SB = dict(MIN_PRICE=10.0, MIN_AVG_VOL_50K=100, MIN_AVG_VOL_200K=200, EXTREME_SALES_PCT=99,
          SALES_GROWTH_PCT=39, MIN_ANNUAL_SALES_M=25, EPS_GROWTH_PCT=39, SALES_GROWTH_EP4=20,
          MKTCAP_SCAN3_B=11, IPO_YEARS_SCAN3=10, ENTRY_MA_PERIOD=70, ATR_MULTIPLE=2.0,
          USE_DI_FILTER=True, USE_TTT_FILTER=False, USE_MWG_FILTER=False, USE_BMS_FILTER=False,
          TI65_BULL=1.05, VOL_9M=9_000_000, PLUS4PCT=4.0, GAP_JUMP_MULT=1.20,
          GAP_RANGE_PCT=0.04)


def sb_safe_pct_rev(new, old):
    if pd.isna(new) or pd.isna(old) or old <= 0: return None
    return ((new - old) / old) * 100


def sb_safe_pct_eps(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0: return None
    return ((new - old) / abs(old)) * 100


def sb_calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def sb_count_gaps(close, high, low, lookback=100):
    prior = close.shift(1)
    is_gap = (prior.notna() & (close > SB["GAP_JUMP_MULT"] * prior)
              & ((high - low) < SB["GAP_RANGE_PCT"] * close)).astype(int)
    return is_gap.rolling(lookback, min_periods=lookback).sum()


def sb_calc_ttt(close, high, low, volume, min_vol=300_000, min_price=10.0,
                max_3bar=1.5, max_today=0.3, lookback=100):
    vol_ok = volume.shift(1).rolling(3).min() >= min_vol
    price_ok = close > min_price
    consol_ok = ((close - close.shift(3)) / close.shift(3) * 100).abs() <= max_3bar
    tight_ok = ((close - close.shift(1)) / close.shift(1) * 100).abs() <= max_today
    no_gaps = sb_count_gaps(close, high, low, lookback) == 0
    return (vol_ok & price_ok & consol_ok & tight_ok & no_gaps).fillna(False)


def sb_calc_mwg(close, high, low, volume, min_vol=100_000, vol_days=3, min_price=10.0,
                max_day=0.4, lookback=100):
    low30 = close.rolling(30).min()
    mom_ok = (close / low30 >= 1.20) | (close.rolling(7).mean() / close.rolling(65).mean() >= 1.05)
    price_ok = close > min_price
    ctrl_ok = ((close - close.shift(1)) / close.shift(1) * 100).abs() <= max_day
    vol_ok = volume.shift(1).rolling(vol_days).min() >= min_vol
    no_gaps = sb_count_gaps(close, high, low, lookback) == 0
    return (mom_ok & price_ok & ctrl_ok & vol_ok & no_gaps).fillna(False)


def sb_calc_bms(close, high, low, volume, open_, min_price=10.0, min_vol_c1=1_000_000,
                min_vol_c2=100_000, breakout_pct=4.0, stability_pct=2.0, close_str_min=0.70):
    candle = ((close - open_) / open_.clip(lower=0.01)) >= 0.02
    today_r = high - low
    stability = ((close.shift(1) / close.shift(2)) - 1).abs() <= (stability_pct / 100)
    cond1 = (candle & (volume > min_vol_c1)
             & (today_r >= (high.shift(1) - low.shift(1)).abs()) & stability)
    cond2 = (((close / close.shift(1)) >= (1 + breakout_pct / 100)) & (volume > volume.shift(1))
             & (volume >= min_vol_c2) & stability)
    cs_ok = ((close - low) / (high - low).clip(lower=1e-9)) >= close_str_min
    return ((cond1 | cond2) & (close >= min_price) & cs_ok).fillna(False)


def stockbee_technical(d):
    """The price/volume half of sb_screen_one(). Returns a dict, or None if filtered / too short.
    d = daily OHLCV, newest last, about 18 months (your code used period='18mo')."""
    try:
        if d is None or len(d) < 70: return None
        close, high, low, volume, open_ = d["Close"], d["High"], d["Low"], d["Volume"], d["Open"]
        last_c = float(close.iloc[-1]); last_v = float(volume.iloc[-1])
        avg50k = float(volume.rolling(50).mean().iloc[-1]) / 1000
        if last_c < SB["MIN_PRICE"] or avg50k < SB["MIN_AVG_VOL_50K"]: return None
        row = dict(price=round(last_c, 2), avg_vol_50k=round(avg50k, 1))
        latr = float(sb_calc_atr(high, low, close, 14).iloc[-1])
        lma = float(close.rolling(SB["ENTRY_MA_PERIOD"]).mean().iloc[-1])
        in_entry_zone = last_c >= lma + SB["ATR_MULTIPLE"] * latr
        row["adr_pct"] = round(float(((high - low) / close * 100).rolling(14).mean().iloc[-1]), 2)
        row["ext_atr"] = round((last_c - float(close.rolling(70).mean().iloc[-1])) / latr, 2) if latr > 0 else None
        ratio = (close.rolling(7).mean() / close.rolling(65).mean()).iloc[-1]
        lti65 = float(ratio) if not pd.isna(ratio) else 0
        row["ti65"] = round(lti65, 3)
        row["flag_ti65"] = lti65 >= SB["TI65_BULL"]
        row["di_bullish"] = last_c > lma
        prior_c = float(close.iloc[-2]); prior_v = float(volume.iloc[-2])
        pct_today = ((last_c - prior_c) / prior_c * 100) if prior_c else 0
        row["flag_9mvol"] = last_v >= SB["VOL_9M"] and last_v > prior_v and last_c >= 1.04 * prior_c
        row["flag_plus4_bo"] = pct_today >= SB["PLUS4PCT"] and last_v > prior_v
        row["flag_minus4_bd"] = pct_today <= -SB["PLUS4PCT"] and last_v > prior_v
        row["ttt_pass"] = bool(sb_calc_ttt(close, high, low, volume).iloc[-1])
        row["mwg_pass"] = bool(sb_calc_mwg(close, high, low, volume).iloc[-1])
        row["bms_pass"] = bool(sb_calc_bms(close, high, low, volume, open_).iloc[-1])
        row["momentum"] = bool(in_entry_zone
                               and (row["di_bullish"] if SB["USE_DI_FILTER"] else True)
                               and (row["ttt_pass"] if SB["USE_TTT_FILTER"] else True)
                               and (row["mwg_pass"] if SB["USE_MWG_FILTER"] else True)
                               and (row["bms_pass"] if SB["USE_BMS_FILTER"] else True))
        return row
    except Exception:
        return None


def stockbee_fundamental(row, fund, relaxed=False):
    """Scans 1-4. fund = dict(rev_q, eps_q, annual_sales_m, market_cap, currency, ipo_years_ago),
    quarters newest first. As in your file the rules are fail-closed: scans 2-4 need six quarters.
    Yahoo's free feed gives at most five, so with relaxed=True a missing sixth quarter is replaced by
    the latest quarter's own growth (CHANGED, opt-in, and the row says so)."""
    revq, epsq = fund.get("rev_q") or [], fund.get("eps_q") or []
    ann, mcap = fund.get("annual_sales_m"), fund.get("market_cap")
    curr, ipo = fund.get("currency"), fund.get("ipo_years_ago")
    price_ok = row["price"] >= SB["MIN_PRICE"]
    avg50k = row["avg_vol_50k"]
    vol100_ok = avg50k >= SB["MIN_AVG_VOL_50K"]; vol200_ok = avg50k >= SB["MIN_AVG_VOL_200K"]
    sales_ok = ann is not None and ann >= SB["MIN_ANNUAL_SALES_M"]
    sg = sb_safe_pct_rev(revq[0], revq[4]) if len(revq) >= 5 else None
    sg1 = sb_safe_pct_rev(revq[1], revq[5]) if len(revq) >= 6 else None
    eg = sb_safe_pct_eps(epsq[0], epsq[4]) if len(epsq) >= 5 else None
    eg1 = sb_safe_pct_eps(epsq[1], epsq[5]) if len(epsq) >= 6 else None
    used_relaxed = False
    if relaxed:
        if sg1 is None and sg is not None: sg1, used_relaxed = sg, True
        if eg1 is None and eg is not None: eg1, used_relaxed = eg, True
    sg2 = (sg + sg1) / 2 if (sg is not None and sg1 is not None) else None
    out = dict(sales_pct=round(sg, 1) if sg is not None else None,
               eps_pct=round(eg, 1) if eg is not None else None,
               annual_sales_m=round(ann, 1) if ann else None,
               mktcap_b=round(mcap / 1e9, 2) if mcap else None, relaxed_used=used_relaxed)
    out["scan1"] = bool(sg is not None and sg >= SB["EXTREME_SALES_PCT"] and sales_ok and price_ok and vol100_ok)
    core = bool(sg is not None and sg >= SB["SALES_GROWTH_PCT"] and sg2 is not None
                and sg2 >= SB["SALES_GROWTH_PCT"] and sales_ok and price_ok and vol100_ok)
    out["scan2"] = core
    usd_ok = (curr == "USD" or curr is None)
    small_mid = mcap is not None and mcap <= SB["MKTCAP_SCAN3_B"] * 1e9 and usd_ok
    new_ipo = ipo is None or ipo <= SB["IPO_YEARS_SCAN3"]
    out["scan3"] = bool(core and small_mid and new_ipo)
    out["scan4"] = bool(eg is not None and eg >= SB["EPS_GROWTH_PCT"]
                        and eg1 is not None and eg1 >= SB["EPS_GROWTH_PCT"]
                        and sg is not None and sg >= SB["SALES_GROWTH_EP4"]
                        and sg2 is not None and sg2 >= SB["SALES_GROWTH_EP4"]
                        and sales_ok and price_ok and vol200_ok)
    return out


# ================================================================= hourly scans
def scan_option_sell(d, h):
    try:
        if len(h) < 4 or len(d) < 21: return False
        h = h.copy(); h["vwap"] = vwap_intraday(h)
        b1 = h.iloc[-2]; b2 = h.iloc[-3]; v1 = _f(h["vwap"].iloc[-2])
        s20 = sma(d["Close"], 20)
        return (b1["Close"] < b1["Open"] and b1["Open"] < b1["High"]
                and b1["Close"] > b1["Low"] and b1["Close"] < b1["High"]
                and b1["Close"] < v1 and b1["Open"] > v1 and b1["High"] > v1
                and b2["Close"] > b2["Open"]
                and b1["Open"] > b2["Open"] and b1["Open"] > b2["Close"]
                and _f(d["Close"].iloc[-1]) < _f(s20.iloc[-1]))
    except Exception:
        return False


def scan_rathod_bullish(d, h):
    try:
        if len(d) < 22 or len(h) < 8: return False
        dc = _f(d["Close"].iloc[-1]); hc = _f(h["Close"].iloc[-1])
        return (dc > _f(sma(d["Close"], 20).iloc[-1])
                and all(dc > _f(d["Low"].iloc[-i]) for i in range(2, 7))
                and _f(rsi_sma(d["Close"]).iloc[-1]) > 40
                and hc > _f(sma(h["Close"], 20).iloc[-1])
                and all(hc > _f(h["Low"].iloc[-i]) for i in range(2, 7))
                and _f(rsi_sma(h["Close"]).iloc[-1]) > 40)
    except Exception:
        return False


def scan_nr7(d):
    try:
        if len(d) < 9: return False
        r0 = _range(d, 0)
        if not all(r0 < _range(d, i) for i in range(1, 8)): return False
        close = _f(d["Close"].iloc[-1]); open_ = _f(d["Open"].iloc[-1])
        prev_vol = _f(d["Volume"].iloc[-2]); today_vol = _f(d["Volume"].iloc[-1])
        if not (close > open_ and close > _f(d["Close"].iloc[-2])): return False
        if not (prev_vol > 10000 and today_vol > prev_vol * 1.25): return False
        s20 = _f(sma(d["Close"], 20).iloc[-1]); s40 = _f(sma(d["Close"], 40).iloc[-1])
        s60 = _f(sma(d["Close"], 60).iloc[-1])
        return s20 > s40 > s60
    except Exception:
        return False


def scan_nr7_weekly_monthly(d, w, m):
    try:
        if not scan_nr7(d): return False
        if w is None or len(w) < 2 or m is None or len(m) < 2: return False
        return (_f(w["Close"].iloc[-1]) > _f(w["Open"].iloc[-1])
                and _f(m["Close"].iloc[-1]) > _f(m["Open"].iloc[-1]))
    except Exception:
        return False


def scan_momentum(d):
    try:
        if len(d) < 52: return False
        close = _f(d["Close"].iloc[-1]); vol = _f(d["Volume"].iloc[-1])
        avg_vol_20 = _f(sma(d["Volume"], 20).iloc[-1])
        s20 = _f(sma(d["Close"], 20).iloc[-1]); s50 = _f(sma(d["Close"], 50).iloc[-1])
        return (close > 50 and vol > avg_vol_20 and s20 > s50
                and _f(rsi_sma(d["Close"], 14).iloc[-1]) > 50 and close > s50
                and close > _f(d["High"].iloc[-2]) and avg_vol_20 > 500000)
    except Exception:
        return False


def scan_dualbb(bars, daily_rsi_val):
    """Dual BB crossover on any intraday timeframe. Returns 'bullish', 'bearish' or None."""
    try:
        if len(bars) < 201: return None
        c = bars["Close"]
        b1b = sma(c, 5); b1d = 0.2611 * c.rolling(5).std()
        b2b = sma(c, 20); b2d = 0.2611 * c.rolling(20).std()
        b1_lo, b1_up = b1b - b1d, b1b + b1d
        b2_lo, b2_up = b2b - b2d, b2b + b2d
        s200 = _f(sma(c, 200).iloc[-1]); close = _f(c.iloc[-1])
        buy = (b1_lo.iloc[-1] > b2_lo.iloc[-1]) and (b1_lo.iloc[-2] <= b2_lo.iloc[-2])
        sell = (b1_up.iloc[-1] < b2_up.iloc[-1]) and (b1_up.iloc[-2] >= b2_up.iloc[-2])
        if buy and close > s200 and daily_rsi_val > 50: return "bullish"
        if sell and close < s200 and daily_rsi_val < 50: return "bearish"
        return None
    except Exception:
        return None


def scan_bullbhai_momentum(d, m15):
    try:
        if len(d) < 5 or len(m15) < 201: return False
        if not (_f(m15["Close"].iloc[-1]) > _f(ema(m15["Close"], 200).iloc[-1])): return False
        if not (_f(d["Close"].iloc[-1]) > _f(d["High"].iloc[-3])): return False
        return _f(rsi_sma(d["Close"], 14).iloc[-1]) > 51
    except Exception:
        return False


def scan_short_term_breakout(d):
    return scan_ep_breakout(d)  # identical rule in both of your files


def scan_potential_breakout(d):
    try:
        if len(d) < 230: return False
        close_today = _f(d["Close"].iloc[-1])
        if not (close_today * 1.05 > _f(d["High"].rolling(200).max().iloc[-1])): return False
        if not (_f(d["High"].rolling(30).max().iloc[-1]) <= _f(d["High"].rolling(8).max().iloc[-31])): return False
        if not (_f(d["Volume"].iloc[-1]) > _f(sma(d["Volume"], 50).iloc[-1])): return False
        return close_today > 90
    except Exception:
        return False


def scan_buy_breakout_100(d):
    """Weekly/monthly alignment is approximated from daily bars, exactly as in your file."""
    try:
        if len(d) < 65: return False
        today_range = _f(d["High"].iloc[-1]) - _f(d["Low"].iloc[-1])
        for i in range(1, 8):
            if not (today_range > _range(d, i)): return False
        if not (_f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-1])): return False
        if not (_f(d["Close"].iloc[-1]) > _f(d["Close"].iloc[-2])): return False
        if not (_f(d["Volume"].iloc[-2]) > 10000): return False
        if not (_f(d["Volume"].iloc[-1]) > _f(d["Volume"].iloc[-2]) * 1.25): return False
        s20 = _f(sma(d["Close"], 20).iloc[-1]); s40 = _f(sma(d["Close"], 40).iloc[-1])
        s60 = _f(sma(d["Close"], 60).iloc[-1])
        if not (s20 > s40 > s60): return False
        if not (_f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-5])): return False
        return _f(d["Close"].iloc[-1]) > _f(d["Open"].iloc[-22])
    except Exception:
        return False


def scan_possible_bottom_out(d, w):
    try:
        if w is None or len(w) < 6: return False
        wh = lambda i: _f(w["High"].iloc[i]); wl = lambda i: _f(w["Low"].iloc[i])
        wc = lambda i: _f(w["Close"].iloc[i]); wo = lambda i: _f(w["Open"].iloc[i])
        for a, b in ((-5, -4), (-4, -3), (-3, -2)):
            if not (wh(a) > wh(b) and wl(a) > wl(b) and wc(a) < wo(a)): return False
        if not (wh(-1) > wh(-2) and wl(-1) > wl(-2) and wc(-1) > wo(-1)): return False
        return wc(-1) >= 100
    except Exception:
        return False


def scan_breaking_day_high_5m(d, m5):
    try:
        if m5 is None or len(m5) < 25: return False
        c, v = m5["Close"], m5["Volume"]
        v_sma20 = v.rolling(20).mean(); max20 = c.shift(1).rolling(20).max()
        if not (c.iloc[-1] > max20.iloc[-1] and v.iloc[-1] > v_sma20.iloc[-1]): return False
        if not (c.iloc[-2] > max20.iloc[-2] and v.iloc[-2] > v_sma20.iloc[-2]): return False
        return _f(d["Volume"].iloc[-1]) > 100_000
    except Exception:
        return False


def scan_option_selection(d, w):
    """Returns 'options_bull', 'options_bear' or False. 'Supertrend' is the ATR lower-band
    approximation from your file."""
    try:
        if len(d) < 20: return False

        def _st(df, period, mult):
            hl2 = (df["High"] + df["Low"]) / 2
            return hl2 - (df["High"] - df["Low"]).rolling(period).mean() * mult

        st7 = _st(d, 7, 3); st10 = _st(d, 10, 2)
        c0 = _f(d["Close"].iloc[-1]); c1 = _f(d["Close"].iloc[-2])
        cross7 = c0 < _f(st7.iloc[-1]) and c1 >= _f(st7.iloc[-2])
        cross10 = c0 < _f(st10.iloc[-1]) and c1 >= _f(st10.iloc[-2])
        if not (cross7 and cross10): return False
        if not (_f(d["Volume"].iloc[-1]) > d["Volume"].iloc[-8:-1].max()): return False
        if w is not None and len(w) >= 15:
            w_rsi = _f(rsi_wilder(w["Close"], 14).iloc[-1])
            if w_rsi >= 55: return "options_bull"
            if w_rsi < 50: return "options_bear"
        return "options_bear"
    except Exception:
        return False


def scan_btst_hourly(d):
    return scan_btst_strong(d)  # identical rule in your hourly and EOD files


# ================================================================= registry
class Ctx:
    """Data for one symbol. Missing frames are None. memo caches rules that feed two lists."""
    def __init__(self, sym, d, w=None, m=None, h=None, h1=None, m15=None, m5=None, mktcap_m=0.0):
        self.sym, self.d, self.w, self.m = sym, d, w, m
        self.h, self.h1, self.m15, self.m5 = h, h1, m15, m5
        self.mktcap_m = mktcap_m
        self.memo = {}


def _daily_rsi(c):
    try: return _f(rsi_sma(c.d["Close"], 14).iloc[-1])
    except Exception: return 50


def _dualbb(c, key, bars):
    if key not in c.memo:
        c.memo[key] = scan_dualbb(bars, _daily_rsi(c)) if bars is not None else None
    return c.memo[key]


def _optsel(c):
    if "opt" not in c.memo: c.memo["opt"] = scan_option_selection(c.d, c.w)
    return c.memo["opt"]


def _need(*names):
    return names


# (key, label, rule in plain words, direction, function, frames the rule needs)
SCANS = {
    "eod": [
        ("high_volume", "High Volume Record", "Volume at a 63-day, 1-year or 8-year high on an up day, with size and liquidity filters", "neutral",
         lambda c: scan_hv(c.d, c.w, c.mktcap_m), ("d", "w")),
        ("weekly_rsi_reversal", "Weekly RSI Reversal", "Weekly RSI above 50, close back above the 10-day average after a red day", "bull",
         lambda c: scan_weekly_rsi_reversal(c.d, c.w), ("d", "w")),
        ("btst_strong", "BTST Strong", "Close above EMA100 and yesterday's high, RSI above 55, volume over 2x its 10-day average", "bull",
         lambda c: scan_btst_strong(c.d), ("d",)),
        ("btst_supertrend", "BTST Supertrend + 52-week high", "Two up closes, above ATR band, within 1% of the 52-week high, RSI above 50", "bull",
         lambda c: scan_btst_supertrend(c.d), ("d",)),
        ("trend_reversal", "Trend Reversal (outside bar)", "Green bar that engulfs the prior red bar's high and low", "bull",
         lambda c: scan_trend_reversal(c.d), ("d",)),
        ("weekly_engulfing", "Weekly Bullish Engulfing", "Green week that engulfs the prior red week", "bull",
         lambda c: scan_weekly_reversal_engulfing(c.d, c.w), ("d", "w")),
    ],
    "stockbee": [
        ("bearish_momentum", "Bearish Momentum", "Close below yesterday's low and the 50-day average, RSI under 50", "bear",
         lambda c: scan_bearish_momentum(c.d), ("d",)),
        ("vcp_setup", "VCP Setup", "Strong 1- or 3-month run above the 200-day average with size", "bull",
         lambda c: scan_vcp(c.d, c.mktcap_m), ("d",)),
        ("satish_bullish", "Satish Bullish", "Green bar above pivot with rising lows, volume above its 10-day average", "bull",
         lambda c: scan_satish_bullish(c.d, c.mktcap_m), ("d",)),
        ("hardik", "Hardik Scan", "Weekly RSI above 50, green bar after a red one, above the 10-day average", "bull",
         lambda c: scan_hardik(c.d, c.w, c.mktcap_m), ("d", "w")),
        ("nr6", "NR6 Narrowest Range", "Today's range is the narrowest of the last 7 bars", "neutral",
         lambda c: scan_nr6(c.d), ("d",)),
        ("rocket_base", "Rocket Base", "Up 20% in 5 days, or 30% in 30 or 90 days, from the low", "bull",
         lambda c: scan_rocket_base(c.d), ("d",)),
        ("ep_breakout", "EP Breakout", "5-day closing high at least 5% above the prior 120-day closing high, on rising volume", "bull",
         lambda c: scan_ep_breakout(c.d), ("d",)),
        ("high_20d", "20-Day High", "Close at or above the 20-day high", "bull",
         lambda c: scan_20day_high(c.d), ("d",)),
        ("high_2month", "2-Month High", "Close at or above the 42-day high", "bull",
         lambda c: scan_2month_high(c.d), ("d",)),
        ("gapup_3pct", "Gap Up 3%", "Open at least 3% above the prior close", "bull",
         lambda c: scan_gapup(c.d), ("d",)),
        ("up_4pct", "Up 4%+", "Close at least 4% above the open", "bull",
         lambda c: scan_up4pct(c.d), ("d",)),
        # Stockbee screens are added by the runner (they use stockbee_technical / stockbee_fundamental)
    ],
    "hourly": [
        ("option_sell", "Option Sell (Rathod)", "Bearish 1-hour reversal at VWAP while daily is below its 20-day average", "bear",
         lambda c: scan_option_sell(c.d, c.h), ("d", "h")),
        ("rathod_bullish", "Rathod Bullish", "Daily and 1-hour both above their 20 averages with rising lows, RSI above 40", "bull",
         lambda c: scan_rathod_bullish(c.d, c.h), ("d", "h")),
        ("nr7", "NR7", "Narrowest range of 8 bars with a green, rising-volume bar, trend stack, green week and month", "bull",
         lambda c: scan_nr7_weekly_monthly(c.d, c.w, c.m), ("d", "w", "m")),  # CHANGED: your call passed m=None, which always failed
        ("momentum", "Momentum", "Above the 50-day average, 20 over 50, RSI above 50, over yesterday's high", "bull",
         lambda c: scan_momentum(c.d), ("d",)),
        ("dualbb_15m_bull", "Dual BB 15m Bullish", "Fast lower band crosses above slow lower band on 15-minute bars, above the 200 average", "bull",
         lambda c: _dualbb(c, "bb15", c.m15) == "bullish", ("d", "m15")),
        ("dualbb_15m_bear", "Dual BB 15m Bearish", "Fast upper band crosses below slow upper band on 15-minute bars, below the 200 average", "bear",
         lambda c: _dualbb(c, "bb15", c.m15) == "bearish", ("d", "m15")),
        ("dualbb_1h_bull", "Dual BB 1h Bullish", "Same crossover on 1-hour bars", "bull",
         lambda c: _dualbb(c, "bb1h", c.h1) == "bullish", ("d", "h1")),
        ("dualbb_1h_bear", "Dual BB 1h Bearish", "Same crossover on 1-hour bars", "bear",
         lambda c: _dualbb(c, "bb1h", c.h1) == "bearish", ("d", "h1")),
        ("bullbhai_momentum", "BullBhai Momentum", "15-minute close above EMA200, daily close above the high of 2 days ago, daily RSI above 51", "bull",
         lambda c: scan_bullbhai_momentum(c.d, c.m15) if c.m15 is not None else False, ("d", "m15")),
        ("short_term_breakout", "Short Term Breakout", "5-day closing high at least 5% above the prior 120-day closing high, on rising volume", "bull",
         lambda c: scan_short_term_breakout(c.d), ("d",)),
        ("potential_breakout", "Potential Breakout", "Within 5% of the 200-day high, contracting 30-day range, volume over its 50-day average", "bull",
         lambda c: scan_potential_breakout(c.d), ("d",)),
        ("buy_breakout_100", "100% Buy Breakout", "Widest range of 8 bars, rising volume, trend stack", "bull",
         lambda c: scan_buy_breakout_100(c.d), ("d",)),
        ("possible_bottom_out", "Possible Bottom Out (weekly)", "Three lower red weeks then a higher green week, close at or above 100", "bull",
         lambda c: scan_possible_bottom_out(c.d, c.w), ("d", "w")),
        ("breaking_5m", "Breaking Day High (5m)", "Two consecutive 5-minute closes above the 20-bar high on above-average volume", "bull",
         lambda c: scan_breaking_day_high_5m(c.d, c.m5), ("d", "m5")),
        ("options_bear", "Options Selection Bearish", "Close crosses below both ATR bands on a volume spike, weak weekly RSI", "bear",
         lambda c: _optsel(c) == "options_bear", ("d", "w")),   # CHANGED: your runner put 'options_bull' results in this list too
        ("options_bull", "Options Selection Bullish", "Same cross while weekly RSI is 55 or higher", "bull",
         lambda c: _optsel(c) == "options_bull", ("d", "w")),
        ("btst_hourly", "BTST (intraday)", "Close above EMA100 and yesterday's high, RSI above 55, volume over 2x average", "bull",
         lambda c: scan_btst_hourly(c.d), ("d",)),
    ],
}
