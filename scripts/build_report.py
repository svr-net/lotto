"""Build results/report.html: results/assimilation.json (written by sim/bin/assimilate) injected into
results/report_template.html. No calculation happens here.

    python scripts/build_report.py
"""
import json


def main():
    data = json.dumps({"assimilation": json.load(open("results/assimilation.json"))}, separators=(",", ":")).replace("</", "<\\/")
    html = open("results/report_template.html").read().replace("/*DATA*/null", data)
    open("results/report.html", "w").write(html)
    print("wrote results/report.html")


if __name__ == "__main__":
    main()
