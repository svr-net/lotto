"""Inject results/walk_forward.json into results/report_template.html -> results/report.html."""
import json, sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/walk_forward.json"
data = json.dumps(json.load(open(src)), separators=(",", ":")).replace("</", "<\\/")
html = open("results/report_template.html").read().replace("/*DATA*/null", data)
open("results/report.html", "w").write(html)
print("wrote results/report.html")
