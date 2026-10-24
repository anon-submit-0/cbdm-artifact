"""
Build data_semeval16/{restaurants,laptops}_reviews.json from the official
SemEval-2016 Task 5 English review-level XML files (train + test), which are
not redistributed here. Place the four XML files in data_semeval16/ first:

  restaurants_train_english.xml  restaurants_test_english.xml
  laptops_train_english.xml      laptops_test_english.xml

Each review becomes {"review_id": "<domain>::<split>::<rid>", "text": sentences
joined by single spaces, "gold": sorted set of ENTITY#ATTRIBUTE categories}.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data_semeval16"

for dom in ("restaurants", "laptops"):
    out = []
    for split in ("train", "test"):
        for rv in ET.parse(D / f"{dom}_{split}_english.xml").getroot().iter("Review"):
            sents = [(s.find("text").text or "").strip() for s in rv.iter("sentence")]
            gold = sorted({o.get("category") for o in rv.iter("Opinion")})
            out.append({"review_id": f"{dom}::{split}::{rv.get('rid')}", "text": " ".join(sents), "gold": gold})
    (D / f"{dom}_reviews.json").write_text(json.dumps(out, ensure_ascii=False))
    print(dom, len(out), "reviews")
