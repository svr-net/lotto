"""Inject results/walk_forward.json (plus results/grammar.json and
results/prime_radix.json, if present) into results/report_template.html -> results/report.html."""
import json, os, sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/walk_forward.json"
grammar = sys.argv[2] if len(sys.argv) > 2 else "results/grammar.json"
payload = json.load(open(src))
if os.path.exists(grammar):
    payload["grammar"] = json.load(open(grammar))
prime_radix = sys.argv[3] if len(sys.argv) > 3 else "results/prime_radix.json"
if os.path.exists(prime_radix):
    payload["prime_radix"] = json.load(open(prime_radix))
data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
html = open("results/report_template.html").read().replace("/*DATA*/null", data)
open("results/report.html", "w").write(html)
print("wrote results/report.html")
