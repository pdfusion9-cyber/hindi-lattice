#!/usr/bin/env python3
"""buniyad.py -- कोडिंग मॉडल की नींव: अंग्रेज़ी वर्णमाला, व्याकरण, शब्द, और कोड-शब्दावली।

  python3 buniyad.py --out store
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="store")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

LOWER = list("abcdefghijklmnopqrstuvwxyz")
UPPER = [c.upper() for c in LOWER]
DIGITS = list("0123456789")
PUNCT = list(".,;:!?'\"()[]{}<>+-*/%=&|^~`@#$\\_")

OPS = ["==", "!=", "<=", ">=", "&&", "||", "**", "//", "->", "=>", ":=", "<<", ">>",
       "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "++", "--", "...", "::", "##"]

ENG = """the be to of and a in that have I it for not on with he as you do at this but his by from
that they we say her she or an will my one all would there their what so up out if about who get which go me
when make can like time no just him know take people into year your good some could them see other than
then now look only come its over think also back after use two how our work first well way even new want
because any these give day most us""".split()

GRAM = {"is": "है", "are": "हैं", "was": "था", "were": "थे", "the": "वह", "a": "एक", "an": "एक",
        "of": "का", "to": "को", "in": "में", "for": "के लिए", "with": "के साथ", "and": "और",
        "or": "या", "not": "नहीं", "if": "अगर", "then": "तब", "else": "अन्यथा", "while": "जबकि",
        "return": "लौटाओ", "def": "परिभाषा", "class": "वर्ग", "import": "आयात", "from": "से",
        "try": "कोशिश", "except": "अपवाद", "finally": "अंत में", "as": "के रूप में",
        "none": "कुछ नहीं", "true": "सत्य", "false": "असत्य"}

KW = """def return if elif else for while break continue pass class import from as try except finally
raise with yield lambda global nonlocal assert del in is not and or None True False async await match case
function var let const new this typeof instanceof switch case default throw catch delete void do
int float double char bool string long short unsigned struct union enum typedef static extern const
volatile sizeof goto NULL printf scanf malloc free include define ifdef endif""".split()

TYPES = """int float str bool list dict set tuple bytes bytearray complex object type range len print input
open read write close append extend insert remove pop sort reverse keys values items get update split join
strip replace format encode decode map filter zip sum min max abs round enumerate sorted reversed any all
self cls args kwargs i j k x y z n m result data value key index count total""".split()

LIBS = """numpy pandas torch tensorflow sklearn matplotlib scipy os sys json re math random time datetime
collections itertools functools pathlib subprocess threading asyncio requests flask django fastapi sqlite3
logging unittest pytest typing dataclasses abc copy pickle csv io""".split()

facts = []
for w in KW:
    facts.append((w, "कोड कीवर्ड", "हाँ"))
for w in TYPES:
    facts.append((w, "कोड नाम", "हाँ"))
for w in LIBS:
    facts.append((w, "पुस्तकालय", "हाँ"))
for e, h in GRAM.items():
    facts.append((e, "अंग्रेज़ी अर्थ", h))
for op in OPS:
    facts.append((op, "कोड चिह्न", "हाँ"))

paths = []
for c in LOWER + UPPER:
    paths.append(f"{c} अंग्रेज़ी अक्षर है।")
for d in DIGITS:
    paths.append(f"{d} एक अंक है।")
for w in ENG:
    paths.append(f"{w} एक अंग्रेज़ी शब्द है।")
for w in KW:
    paths.append(f"{w} एक कोड कीवर्ड है।")
for w in TYPES:
    paths.append(f"{w} कोड में इस्तेमाल होता है।")
for w in LIBS:
    paths.append(f"{w} एक पुस्तकालय है।")
for op in OPS:
    paths.append(f"{op} एक कोड चिह्न है।")

with open(f"{a.out}/gyaan_buniyad.txt", "w", encoding="utf-8") as f:
    for x, r, y in facts:
        f.write(f"{x} | {r} | {y}\n")
with open(f"{a.out}/path_buniyad.txt", "w", encoding="utf-8") as f:
    for p in paths:
        f.write(p + "\n")

print(f"अक्षर {len(LOWER)+len(UPPER)} | अंक {len(DIGITS)} | चिह्न {len(PUNCT)}")
print(f"अंग्रेज़ी शब्द {len(ENG)} | कोड कीवर्ड {len(KW)} | कोड नाम {len(TYPES)} | पुस्तकालय {len(LIBS)}")
print(f"नींव-तथ्य {len(facts):,} | नींव-पाथ {len(paths):,}")
print(f"सहेजा: {a.out}/gyaan_buniyad.txt, {a.out}/path_buniyad.txt")
