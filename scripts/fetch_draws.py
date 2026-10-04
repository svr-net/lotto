"""Fetch UK Lotto draws from lotteryresultsfeed.com and cache them under data/.

Usage: LRF_KEY=<token> python scripts/fetch_draws.py [--slug uk-lotto] [--country uk]
Writes data/<country>_<slug>_raw.json and .csv (date,round,n1..n6,bonus).
Note: the free API tier only returns the last month of results.
"""
import argparse, csv, json, os, sys, time
import urllib.error, urllib.parse, urllib.request

BASE = "https://www.lotteryresultsfeed.com/api/lottery"


def get(path, params, key, tries=8):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0", "Accept": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 429, 500, 502, 503) and i < tries - 1:
                time.sleep(2 * (i + 1))
                continue
            raise SystemExit(f"HTTP {e.code} for {url}: {e.read()[:300]!r}")
    raise SystemExit("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="lotto")
    ap.add_argument("--country", default="uk")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    key = os.environ.get("LRF_KEY") or sys.exit("LRF_KEY not set")
    lots = get("lotteries", {"country": a.country}, key)["lotteries"]
    print("available:", [(l["id"], l["slug"]) for l in lots])
    lot = next(l for l in lots if l["slug"] == a.slug)
    draws, year = {}, int(time.strftime("%Y"))
    empty = 0
    while year >= 1994 and empty < 2:
        res = get("results", {"id": lot["id"], "year": year}, key)
        rows = res.get("results", res) if isinstance(res, dict) else res
        print(year, len(rows))
        empty = empty + 1 if not rows else 0
        for r in rows:
            draws[(r["draw_date"], r.get("draw_type") or "")] = r
        year -= 1
    ordered = [draws[d] for d in sorted(draws)]  # sorted by (date, round)
    json.dump(ordered, open(os.path.join(a.out, f"{a.country}_{a.slug}_raw.json"), "w"))
    with open(os.path.join(a.out, f"{a.country}_{a.slug}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
        for r in ordered:
            w.writerow([r["draw_date"], r.get("draw_type") or "", *r["balls"], r.get("ball_bonus")])
    print("draws:", len(ordered), ordered[0]["draw_date"], "->", ordered[-1]["draw_date"])


if __name__ == "__main__":
    main()
