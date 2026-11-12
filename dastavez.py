#!/usr/bin/env python3
"""dastavez.py -- दस्तावेज़ कार्य: पढ़ना, सुधारना, समझाना, बनाना, अनुवाद, संपादन।

  python3 dastavez.py --kaam padho   --file doc.txt
  python3 dastavez.py --kaam sudharo --file doc.txt
  python3 dastavez.py --kaam samjhao --file doc.txt
  python3 dastavez.py --kaam banao   --file naya.txt --vishay "रिपोर्ट"
  python3 dastavez.py --kaam anuvaad --file doc.txt --se hi --tak en
"""
import argparse, os, re, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--kaam", required=True,
                choices=["padho", "sudharo", "samjhao", "banao", "anuvaad", "badlo", "jodo"])
ap.add_argument("--file", required=True)
ap.add_argument("--vishay", default="")
ap.add_argument("--se", default="hi")
ap.add_argument("--tak", default="en")
ap.add_argument("--dhoondo", default="")
ap.add_argument("--badle", default="")
a = ap.parse_args()

def padho(p):
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return open(p, encoding=enc).read()
        except Exception:
            continue
    return ""

def vaakya(t):
    return [s.strip() for s in re.split(r"[।\n]+", t) if s.strip()]

def sabd(t):
    return re.findall(r"[^\s।,?!;:()\[\]{}\"']+", t)

# ---------- 1. पढ़ना ----------
if a.kaam == "padho":
    t = padho(a.file)
    v = vaakya(t)
    w = sabd(t)
    print(f"फ़ाइल: {a.file}")
    print(f"  अक्षर {len(t):,} | शब्द {len(w):,} | वाक्य {len(v):,} | पैराग्राफ {t.count(chr(10)+chr(10))+1}")
    print(f"  अलग शब्द {len(set(w)):,} | औसत वाक्य-लंबाई {len(w)/max(len(v),1):.1f}")
    ankh = Counter(ch for ch in t if "\u0900" <= ch <= "\u097F")
    print(f"  देवनागरी अक्षर {len(ankh)} अलग")
    print(f"\nपहला वाक्य: {v[0][:100] if v else '—'}")
    print(f"आख़िरी वाक्य: {v[-1][:100] if v else '—'}")

# ---------- 2. सुधारना ----------
elif a.kaam == "sudharo":
    t = padho(a.file)
    badle = 0
    t2 = t
    # दोहरा स्थान
    t2, n = re.subn(r"[ \t]{2,}", " ", t2); badle += n
    # दंड से पहले स्थान
    t2, n = re.subn(r"\s+।", "।", t2); badle += n
    # दोहरा दंड
    t2, n = re.subn(r"।{2,}", "।", t2); badle += n
    # वाक्य के बाद स्थान
    t2, n = re.subn(r"।(\S)", r"। \1", t2); badle += n
    # दोहरे शब्द
    t2, n = re.subn(r"\b(\S+)\s+\1\b", r"\1", t2); badle += n
    # खाली पंक्तियाँ
    t2, n = re.subn(r"\n{3,}", "\n\n", t2); badle += n
    out = a.file.replace(".txt", "_sudhra.txt")
    open(out, "w", encoding="utf-8").write(t2)
    print(f"सुधार: {badle} जगह बदली")
    print(f"सहेजा: {out}")
    print(f"  पहले {len(t):,} अक्षर -> अब {len(t2):,}")

# ---------- 3. समझाना ----------
elif a.kaam == "samjhao":
    t = padho(a.file)
    v = vaakya(t)
    w = sabd(t)
    f = Counter(x for x in w if len(x) >= 4 and x not in
                {"है","हैं","था","थी","थे","और","का","की","के","में","से","को","पर","यह","वह","इस","उस"})
    print(f"दस्तावेज़: {a.file}")
    print(f"  {len(v):,} वाक्य, {len(w):,} शब्द")
    print(f"\nमुख्य विषय (सबसे ज़्यादा आने वाले शब्द):")
    for x, c in f.most_common(10):
        print(f"   {x} — {c} बार")
    print(f"\nसार (पहला और आख़िरी):")
    print(f"   {v[0][:150] if v else '—'}…")
    print(f"   …{v[-1][:150] if v else '—'}")
    # सरल भाषा में
    print(f"\nसरल शब्दों में: यह दस्तावेज़ {len(v)} वाक्यों में मुख्यतः "
          f"'{f.most_common(1)[0][0] if f else '—'}' के बारे में है, "
          f"और इसमें {len(set(w))} अलग शब्द इस्तेमाल हुए हैं।")

# ---------- 4. बनाना ----------
elif a.kaam == "banao":
    vishay = a.vishay or "विषय"
    tpl = f"""# {vishay}

## परिचय
यह दस्तावेज़ {vishay} पर आधारित है।

## मुख्य बिंदु
1. पहला बिंदु — {vishay} का परिचय।
2. दूसरा बिंदु — इसके लाभ और उपयोग।
3. तीसरा बिंदु — आगे की दिशा।

## निष्कर्ष
{vishay} पर यह संक्षिप्त टिप्पणी है।
"""
    open(a.file, "w", encoding="utf-8").write(tpl)
    print(f"बनाया: {a.file} ({len(tpl)} अक्षर)")
    print(f"  विषय: {vishay} | खंड: 3 | वाक्य: {len(vaakya(tpl))}")

# ---------- 5. अनुवाद (शब्दकोश-आधारित) ----------
elif a.kaam == "anuvaad":
    t = padho(a.file)
    # छोटा द्विभाषी शब्दकोश (विस्तार योग्य)
    HI_EN = {"है":"is","हैं":"are","था":"was","थे":"were","और":"and","का":"of","की":"of","के":"of",
             "में":"in","से":"from","को":"to","पर":"on","यह":"this","वह":"that","नहीं":"not",
             "देश":"country","राजधानी":"capital","नदी":"river","पर्वत":"mountain","भाषा":"language",
             "जनसंख्या":"population","क्षेत्रफल":"area","शहर":"city","गाँव":"village","फिल्म":"film",
             "पुस्तक":"book","वर्ष":"year","जल":"water","आकाश":"sky","सूर्य":"sun","चंद्रमा":"moon"}
    w = sabd(t)
    out = []
    hit = 0
    for x in w:
        y = HI_EN.get(x)
        if y:
            out.append(y); hit += 1
        else:
            out.append(x)
    print(f"अनुवाद ({a.se} -> {a.tak}): {len(w)} शब्द")
    print(f"  शब्दकोश में मिले: {hit} ({hit/max(len(w),1)*100:.1f}%)")
    print(f"\n{' '.join(out)[:400]}")
    if hit / max(len(w), 1) < 0.5:
        print(f"\n  (सिर्फ़ {hit} शब्द मिले -- पूरा अनुवाद नहीं, शब्दकोश छोटा है)")

# ---------- 6. बदलना (खोज-प्रतिस्थापन) ----------
elif a.kaam == "badlo":
    t = padho(a.file)
    if not a.dhoondo:
        print("--dhoondo दें"); sys.exit(1)
    n = t.count(a.dhoondo)
    t2 = t.replace(a.dhoondo, a.badle)
    open(a.file, "w", encoding="utf-8").write(t2)
    print(f"बदला: '{a.dhoondo}' -> '{a.badle}' | {n} जगह")

# ---------- 7. जोड़ना ----------
elif a.kaam == "jodo":
    t = padho(a.file)
    extra = a.vishay
    open(a.file, "a", encoding="utf-8").write("\n" + extra + "\n")
    print(f"जोड़ा: {len(extra)} अक्षर | अब कुल {len(t)+len(extra):,}")
