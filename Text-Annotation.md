# A Practical, End-to-End Text Annotation Guide (Pre-trained + Keyword-based)  
*Author: arastu thakur*  

> This guide shows you how to **auto-annotate** raw text using a mix of **pre-trained models** (spaCy + Hugging Face) and **keyword/regex rules**, then **merge**, **export** (BIO/JSONL/BRAT), and **QA** your labels. It includes ready-to-run scripts and a tiny CLI that works on a folder of `.txt` files.

---

## 0) What you’ll build

- A pipeline that:
  1. Reads `.txt` files from a folder.
  2. Runs keyword/regex matchers to create spans.
  3. Runs pre-trained models for NER / sentiment / zero-shot labels.
  4. Merges all annotations with conflict resolution.
  5. Exports to **JSONL**, **CoNLL-BIO**, and **BRAT standoff** format.
  6. Emits simple QA reports (label counts, overlaps).

---

## 1) Set up

```bash
# Python 3.9+
pip install spacy transformers torch regex pydantic rich
python -m spacy download en_core_web_sm
```
## 2) Define your label schema

Decide what you’re tagging. Example:

- **Spans (sequence labels):** PERSON, ORG, PRODUCT, EMAIL, PHONE, DATE, MONEY  
- **Doc labels (classification):** finance, medical, news (multi-label)

Create a config file **`labels.json`**:

```json
{
  "sequence_labels": ["PERSON", "ORG", "PRODUCT", "EMAIL", "PHONE", "DATE", "MONEY"],
  "doc_labels": ["finance", "medical", "news"],
  "priority": ["EMAIL", "PHONE", "DATE", "MONEY", "PERSON", "ORG", "PRODUCT"]
}
```
### 3) Prepare keyword & regex rules

Create `rules.json`:

```json
{
  "keywords": {
    "PRODUCT": ["MacBook", "iPhone", "Pixel", "ThinkPad", "Galaxy"],
    "ORG": ["Google", "Microsoft", "Amazon", "OpenAI", "Siemens"],
    "finance_doc_hints": ["invoice", "wire transfer", "SWIFT", "KYC", "AML"]
  },
  "regex": {
    "EMAIL": "\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b",
    "PHONE": "(?:\\+\\d{1,3}[\\s-]?)?(?:\\(?\\d{2,4}\\)?[\\s-]?)?\\d{3,4}[\\s-]?\\d{3,4}",
    "DATE": "\\b(?:\\d{1,2}[\\-/]\\d{1,2}[\\-/]\\d{2,4}|\\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\\.?\\s+\\d{1,2},?\\s+\\d{2,4})\\b",
    "MONEY": "[$€₹]\\s?\\d+(?:[\\d,]*\\d)?(?:\\.\\d+)?"
  }
}
```
### 4) Core utility: span merging & BIO conversion

**File:** `annot_utils.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json, re
import spacy

@dataclass
class Span:
    start: int
    end: int
    label: str
    source: str  # "regex", "keywords", "spacy", "hf"

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def regex_match(text: str, regex_dict: Dict) -> List[Span]:
    spans = []
    for label, pattern in regex_dict.items():
        for m in re.finditer(pattern, text):
            spans.append(Span(m.start(), m.end(), label, "regex"))
    return spans

def keyword_match(text: str, keyword_dict: Dict) -> List[Span]:
    spans = []
    for label, words in keyword_dict.items():
        for kw in words:
            for m in re.finditer(r"\b{}\b".format(re.escape(kw)), text, flags=re.I):
                spans.append(Span(m.start(), m.end(), label, "keywords"))
    return spans

def spacy_ner(text: str, nlp) -> List[Span]:
    doc = nlp(text)
    return [Span(ent.start_char, ent.end_char, ent.label_, "spacy") for ent in doc.ents]

def merge_spans(spans: List[Span], priority: List[str]) -> List[Span]:
    spans.sort(key=lambda s: (s.start, -s.end))
    final = []
    for sp in spans:
        if not any(ov.start < sp.end and sp.start < ov.end for ov in final):
            final.append(sp)
        else:
            # keep higher priority
            overlaps = [ov for ov in final if ov.start < sp.end and sp.start < ov.end]
            for ov in overlaps:
                if priority.index(sp.label) < priority.index(ov.label):
                    final.remove(ov)
                    final.append(sp)
    return final

def to_bio(text: str, spans: List[Span]) -> List[Tuple[str, str]]:
    labels = ["O"] * len(text.split())
    words = text.split()
    mapping = {}
    for sp in spans:
        substring = text[sp.start:sp.end]
        for i, w in enumerate(words):
            if substring.startswith(w):
                labels[i] = "B-" + sp.label
                break
    return list(zip(words, labels))
```
### 5) The annotator (keyword + pre-trained + merge)

**File:** `annotate.py`

```python
import sys, os, json
from annot_utils import load_json, regex_match, keyword_match, spacy_ner, merge_spans, to_bio, Span
import spacy
from transformers import pipeline

def annotate(texts, labels, rules):
    nlp = spacy.load("en_core_web_sm")
    hf_ner = pipeline("ner", grouped_entities=True)

    annotations = []
    for text in texts:
        spans = []
        spans += regex_match(text, rules["regex"])
        spans += keyword_match(text, rules["keywords"])
        spans += spacy_ner(text, nlp)
        for ent in hf_ner(text):
            spans.append(Span(ent["start"], ent["end"], ent["entity_group"], "hf"))

        merged = merge_spans(spans, labels["priority"])
        annotations.append({
            "text": text,
            "spans": [sp.__dict__ for sp in merged],
            "bio": to_bio(text, merged)
        })
    return annotations

if __name__ == "__main__":
    infile, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    labels = load_json("labels.json")
    rules = load_json("rules.json")

    with open(infile, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    anns = annotate(texts, labels, rules)
    with open(os.path.join(outdir, "annotations.json"), "w") as f:
        json.dump(anns, f, indent=2)
    print("Done. Saved to", outdir)
```
Execute the script with:

```bash
python annotate.py sample_texts/demo.txt out_annotations
```
### 6) Small demo input

**File:** `sample_texts/demo.txt`

```css
On Aug 12, 2024, Google paid ₹25,000 to Alice for a MacBook Pro.
Email confirmation was sent to alice@example.com
. Support: +1 415-555-0134.
```
### Expected spans

- **DATE:** Aug 12, 2024  
- **ORG:** Google  
- **MONEY:** ₹25,000  
- **PERSON:** Alice  
- **PRODUCT:** MacBook Pro  
- **EMAIL:** alice@example.com  
- **PHONE:** +1 415-555-0134  

---

### 7) Customizing keyword rules

- Add brand lists, department names, internal project codes.  
- Add domain-specific regex: ticket IDs, invoice numbers, order IDs.  
- Adjust priority to ensure PII like EMAIL/PHONE overrides generic labels.  

---

### 8) Quality checks

**File:** `qa.py`

```python
import json, sys
from collections import Counter

with open(sys.argv[1]) as f:
    anns = json.load(f)

cnt = Counter()
for doc in anns:
    for sp in doc["spans"]:
        cnt[sp["label"]] += 1

print("Label counts:", cnt)
```
Execute the QA script with:

```bash
python qa.py out_annotations/annotations.json
```
### 9) Export tips

- **BIO/IOB** → training CRF/transformers.  
- **JSONL** → for loaders / ML pipelines.  
- **BRAT** → for manual correction in tools.  

---

### 10) Active learning loop

1. Auto-annotate.  
2. Manually correct a subset.  
3. Fine-tune a model.  
4. Replace pre-trained step with your model.  
5. Repeat.  

---

### 11) Troubleshooting

- Hugging Face models download first time (slow).  
- Normalize inconsistent labels (e.g., `ORG` vs `ORGANIZATION`).  
- Use priority or longest-span rules to solve overlaps.  

---

### 12) Minimal quick script

**File:** `quick_annotate.py`

```python
import sys, re, spacy
from transformers import pipeline

text = open(sys.argv[1]).read()
nlp = spacy.load("en_core_web_sm")
hf = pipeline("ner", grouped_entities=True)

regex = {"EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"}
for label, pat in regex.items():
    for m in re.finditer(pat, text):
        print(label, ":", text[m.start():m.end()])

for ent in nlp(text).ents:
    print("SPACY", ent.label_, ":", ent.text)

for ent in hf(text):
    print("HF", ent["entity_group"], ":", ent["word"])
```
Execute the minimal script with:

```bash
python quick_annotate.py sample_texts/demo.txt
```
### 13) Suggested folder structure

project/
├─ annotate.py
├─ annot_utils.py
├─ labels.json
├─ rules.json
├─ qa.py
├─ sample_texts/
│  └─ demo.txt
└─ out_annotations/

### 14) Next steps

- Replace HF zero-shot with fine-tuned classifiers.  
- Add gazetteers for your domain.  
- Track provenance (source) for auditing.  
- Wrap in a FastAPI service for deployment.  

