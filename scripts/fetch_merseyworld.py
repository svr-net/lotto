"""Fetch the full UK Lotto draw history from lottery.merseyworld.com.

Usage: python scripts/fetch_merseyworld.py [--out data/uk_lotto_full.csv] [--html saved.html]

The merseyworld "CSV" export is an HTML page wrapping a <PRE> block of
comma-separated rows (newest first), one row per ball-drawing round:

    No., Day,DD,MMM,YYYY, N1,N2,N3,N4,N5,N6,BN,   Jackpot, Wins,   Machine  ,Set

Balls are listed in draw order. This writes a chronological CSV with the
main numbers sorted ascending (n1..n6) and in the order drawn (d1..d6):

    draw_no,date,round,n1,n2,n3,n4,n5,n6,bonus,jackpot,wins,machine,ball_set,d1,d2,d3,d4,d5,d6

Draws held as two rounds (same draw number) get round 1 / 2 in feed order.
No authentication is required and the export covers every draw since 1994.
"""
import argparse, csv, re, sys, urllib.request
from datetime import datetime

URL = ("http://lottery.merseyworld.com/cgi-bin/lottery?days=2&Machine=Z&Ballset=0"
       "&order=0&show=1&year=0&display=CSV")
ROW = re.compile(r"^\s*(\d+),\s*\w+,\s*(\d+),\s*(\w+),\s*(\d{4}),((?:\s*\d+,){7})\s*(\d+),\s*(\d+),\s*([^,]*?)\s*,\s*(\S*)\s*$")


def fetch(url=URL):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("latin-1")


def parse(text):
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        no, dd, mon, yyyy, balls, jackpot, wins, machine, ball_set = m.groups()
        nums = [int(x) for x in balls.replace(" ", "").strip(",").split(",")]
        date = datetime.strptime(f"{dd} {mon} {yyyy}", "%d %b %Y").date().isoformat()
        rows.append(dict(draw_no=int(no), date=date, main=sorted(nums[:6]), drawn=nums[:6], bonus=nums[6],
                         jackpot=int(jackpot), wins=int(wins), machine=machine, ball_set=ball_set))
    # Feed is newest first; within a two-round draw, the first listed row is round 1.
    seen = {}
    for r in rows:
        seen[r["draw_no"]] = seen.get(r["draw_no"], 0) + 1
        r["round"] = seen[r["draw_no"]]
    rows.sort(key=lambda r: (r["draw_no"], r["round"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/uk_lotto_full.csv")
    ap.add_argument("--html", help="parse a previously saved page instead of downloading")
    a = ap.parse_args()
    text = open(a.html, encoding="latin-1").read() if a.html else fetch()
    rows = parse(text)
    if not rows:
        sys.exit("no draw rows found in feed")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_no", "date", "round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
                    "jackpot", "wins", "machine", "ball_set", "d1", "d2", "d3", "d4", "d5", "d6"])
        for r in rows:
            w.writerow([r["draw_no"], r["date"], r["round"], *r["main"], r["bonus"],
                        r["jackpot"], r["wins"], r["machine"], r["ball_set"], *r["drawn"]])
    print(f"wrote {len(rows)} rows ({rows[0]['date']} -> {rows[-1]['date']}) to {a.out}")


if __name__ == "__main__":
    main()
