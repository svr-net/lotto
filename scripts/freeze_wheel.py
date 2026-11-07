"""Freeze a wheel's lines so the pipeline never changes them, and export them for entering in the app.

Run once to freeze (it refuses to overwrite an existing freeze unless --replace is given):

    python scripts/freeze_wheel.py --first-draw 2026-11-07

It copies the consensus "3 if 4" wheel from results/consensus_wheels.json into results/frozen_wheel.json
(lines, numbers, odds, the date frozen and the first draw it plays), and writes the exports the report
links to: results/exports/frozen_wheel_app.txt (one line per board, numbers in ascending order, as the
app asks for them) and results/exports/frozen_wheel.csv. score_live.py scores the frozen lines against
every draw from the first one on. Re-run with --export-only to rewrite the exports from the freeze.
"""
import argparse, datetime, json, os

FROZEN = "results/frozen_wheel.json"


def write_exports(fz: dict, folder: str = "results/exports") -> None:
    os.makedirs(folder, exist_ok=True)
    lines = fz["lines"]
    head = (f"UK Lotto: frozen \"{fz['wheel']}\" wheel on {' '.join(map(str, fz['numbers']))}\n"
            f"{len(lines)} lines, £{len(lines) * fz['price']:.0f} a draw (£{fz['price']:.0f} a line), playing every draw from {fz['first_draw']}.\n"
            f"Guarantee: whenever 4 of the six drawn numbers are among these 15, at least one line matches 3.\n\n")
    with open(f"{folder}/frozen_wheel_app.txt", "w") as fh:
        fh.write(head + "".join(f"Line {i + 1:2d}:  {'  '.join(f'{n:2d}' for n in sorted(l))}\n" for i, l in enumerate(lines)))
    with open(f"{folder}/frozen_wheel.csv", "w") as fh:
        fh.write("line,n1,n2,n3,n4,n5,n6\n" + "".join(f"{i + 1},{','.join(map(str, sorted(l)))}\n" for i, l in enumerate(lines)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default="3 if 4")
    ap.add_argument("--first-draw", help="date of the first draw the frozen lines play (YYYY-MM-DD)")
    ap.add_argument("--replace", action="store_true", help="replace an existing freeze")
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    if a.export_only:
        write_exports(json.load(open(FROZEN)))
        return
    if os.path.exists(FROZEN) and not a.replace:
        raise SystemExit(f"{FROZEN} exists: the lines are frozen. Use --replace to change them.")
    c = json.load(open("results/consensus_wheels.json"))
    w = c["wheels"][a.wheel]
    fz = {"wheel": a.wheel, "numbers": c["consensus"], "lines": [sorted(l) for l in w["lines"]], "price": c["price"],
          "frozen_on": datetime.date.today().isoformat(), "first_draw": a.first_draw or c["next_draw_date"],
          "guarantee": "3 if 4", "p": w["p"], "fitted": w["fitted"], "fixed_prizes": w["fixed_prizes"],
          "source": f"consensus 15 of results/consensus_wheels.json, history to {c['last_date']}"}
    with open(FROZEN, "w") as fh:
        json.dump(fz, fh, indent=1)
    write_exports(fz)
    print(f"froze {len(fz['lines'])} lines on {fz['numbers']}, first draw {fz['first_draw']}")


if __name__ == "__main__":
    main()
