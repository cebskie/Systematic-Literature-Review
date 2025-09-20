#!/usr/bin/env python3
"""
auto_screening_rag.py
Automasi: search (Crossref), cek OA (Unpaywall), download PDF, ekstrak teks, screening (rule + semantic),
export .bib, screening_log.csv, extracted_data.csv, prisma.png.

NOTE: Sesuaikan CONFIG di bawah.
"""

import os
import sys
import time
import json
import math
import requests
import bibtexparser
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CONFIG = {
    "mailto": "salmaaceibaabdillah@mail.ugm.ac.id",   # gunakan emailmu untuk Crossref/Unpaywall
    "unpaywall_email": "salmaaceibaabdillah@mail.ugm.ac.id",
    "query": '("sentiment" OR "opinion" OR "polarity") AND ("machine learning" OR “naive bayes” OR “random forest” OR “SVM”) AND ("cross domain" OR "multi domain" OR "domain adaptation" OR "domain shift" OR "domain invariant" OR "out-of-domain")',  # default search string
    "filters": {
        "from_pub_date": 2020,
        "until_pub_date": 2025,
        "has_full_text": True,   # prefer records with links
    },
    "max_results": 200,   # limit Crossref results to save time
    "output_dir": "screening_output",
    "semantic_model": "all-MiniLM-L6-v2",  # sentence-transformers model
    "similarity_threshold": 0.55,  # semantic include threshold (0-1)
    "rule_includes" : [
        "machine learning algorithm",
        "sentiment classification",
        "peer-reviewed",
        "journal article",
        "conference paper",
        "English",
        "multi-domain",
        "cross-domain",
        "accuracy",
        "f1-score",
        "quantitative results"
    ],

    "rule_excludes" : [
        "deep learning",
        "neural network",
        "transformer",
        "review paper",
        "survey",
        "duplicate",
        "single domain",
        "no quantitative results",
    ],

    "download_pdfs": True,
}
# ----------------------------

OUT = Path(CONFIG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
PDF_DIR = OUT / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2/"  # append DOI + ?email=

HEADERS = {"User-Agent": f"AutoScreening/1.0 (mailto:{CONFIG['mailto']})"}

# load embedding model
print("Loading embedding model...")
model = SentenceTransformer(CONFIG["semantic_model"])

def crossref_search(query, max_results=100, from_year=None, until_year=None):
    rows = []
    cursor = None
    per_page = 100
    retrieved = 0
    params = {
        "query.bibliographic": query,
        "rows": min(per_page, max_results),
        "mailto": CONFIG["mailto"],
    }
    if from_year:
        params["filter"] = f"from-pub-date:{from_year}"
    if until_year:
        f = params.get("filter", "")
        if f:
            f += f",until-pub-date:{until_year}"
        else:
            f = f"until-pub-date:{until_year}"
        params["filter"] = f

    r = requests.get(CROSSREF_API, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("message", {})
    items = data.get("items", [])
    for it in items:
        rows.append(it)
    return rows

def call_unpaywall(doi):
    url = UNPAYWALL_API + doi
    params = {"email": CONFIG["unpaywall_email"]}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 200:
        return r.json()
    return None

def get_best_pdf_link(unpaywall_json):
    # Unpaywall gives 'best_oa_location' with url_for_pdf
    if not unpaywall_json:
        return None
    bo = unpaywall_json.get("best_oa_location")
    if bo:
        pdf = bo.get("url_for_pdf") or bo.get("url")
        return pdf
    # fallback: look in oa_locations
    for loc in unpaywall_json.get("oa_locations", []):
        p = loc.get("url_for_pdf") or loc.get("url")
        if p:
            return p
    return None

def download_pdf(url, outpath):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
            with open(outpath, "wb") as f:
                f.write(r.content)
            return True
        # sometimes content-type is html but contains pdf link redirect - try to write anyway
        if r.status_code == 200:
            with open(outpath, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        # ignore download errors
        return False
    return False

def extract_text_from_pdf(path):
    try:
        txt = extract_text(str(path))
        return txt
    except Exception as e:
        return ""

def brief_metadata_from_crossref(item):
    doi = item.get("DOI", "")
    title = " ".join(item.get("title", []))
    abstract = item.get("abstract", "") or ""
    authors = []
    for a in item.get("author", []):
        nm = []
        if a.get("given"):
            nm.append(a["given"])
        if a.get("family"):
            nm.append(a["family"])
        authors.append(" ".join(nm))
    year = None
    if item.get("issued") and item["issued"].get("date-parts"):
        year = item["issued"]["date-parts"][0][0]
    journal = item.get("container-title", [""])[0]
    return {"doi": doi, "title": title, "abstract": abstract, "authors": "; ".join(authors), "year": year, "journal": journal}

def rule_based_screen(title, abstract):
    txt = (title + " " + abstract).lower()
    for ex in CONFIG["rule_excludes"]:
        if ex.lower() in txt:
            return False, f"excluded rule: {ex}"
    for inc in CONFIG["rule_includes"]:
        if inc.lower() in txt:
            return True, f"matched include keyword: {inc}"
    return None, "no rule match"  # undecided by rules

def semantic_screen(title, abstract, rqs):
    # rqs: list of research question strings
    combined = (title + " " + abstract).strip()
    if not combined:
        return 0.0
    emb_doc = model.encode(combined, convert_to_tensor=True)
    # compute max similarity to any rq
    max_sim = 0.0
    best_rq = None
    for rq in rqs:
        emb_rq = model.encode(rq, convert_to_tensor=True)
        sim = util.cos_sim(emb_doc, emb_rq).item()
        if sim > max_sim:
            max_sim = sim
            best_rq = rq
    decision = max_sim >= CONFIG["similarity_threshold"]
    return max_sim, best_rq, decision

def export_bib_from_crossref_items(items, outpath):
    # Create minimal bibtex entries
    bib_entries = []
    for it in items:
        rec = {
            "ENTRYTYPE": "article",
            "ID": it.get("DOI", "").replace("/", "_"),
            "title": " ".join(it.get("title", [])),
            "author": " and ".join([" ".join(filter(None, [a.get("given",""), a.get("family","")])) for a in it.get("author", [])]) if it.get("author") else "",
            "year": str(it.get("issued", {}).get("date-parts", [[""]])[0][0]) if it.get("issued") else "",
            "doi": it.get("DOI", ""),
            "journal": it.get("container-title", [""])[0] if it.get("container-title") else ""
        }
        bib_entries.append(rec)
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = bib_entries
    writer = bibtexparser.bwriter.BibTexWriter()
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(writer.write(db))
    return outpath

def make_prisma(counts, out_png):
    """
    counts: dict with keys:
      total_records, duplicates_removed, screened_title_abstract, excluded_title_abstract,
      full_text_assessed, excluded_full_text, included_final
    """
    # simple visual using matplotlib boxes/arrows
    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis("off")
    # positions
    boxes = [
        ("Records identified\n(from databases)", counts["total_records"], (0.5, 0.95)),
        ("Records after\nduplicates removed", counts["total_records"]-counts["duplicates_removed"], (0.5, 0.85)),
        ("Records screened\n(title/abstract)", counts["screened_title_abstract"], (0.5, 0.65)),
        ("Records excluded\n(title/abstract)", counts["excluded_title_abstract"], (0.5, 0.45)),
        ("Full-text articles\nassessed for eligibility", counts["full_text_assessed"], (0.5, 0.25)),
        ("Full-text articles\nexcluded (with reasons)", counts["excluded_full_text"], (0.5, 0.12)),
        ("Studies included\nin qualitative synthesis", counts["included_final"], (0.5, 0.02)),
    ]
    for text, n, (x,y) in boxes:
        ax.text(x, y, f"{text}\n\nn = {n}", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    # arrows
    ax.annotate("", xy=(0.5,0.88), xytext=(0.5,0.92), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5,0.78), xytext=(0.5,0.82), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5,0.58), xytext=(0.5,0.68), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5,0.38), xytext=(0.5,0.48), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5,0.2), xytext=(0.5,0.22), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5,0.07), xytext=(0.5,0.1), arrowprops=dict(arrowstyle="->"))
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png

def main():
    # Load RQs from protocol or allow override:
    # default RQs from Assignment 2 (document)
    RQS = [
        "What is the purpose of applying machine learning methods in multi-domain sentiment analysis?",
        "What machine learning models are used for multi-domain sentiment analysis?",
        "How is the performance comparison of various machine learning models for multi-domain sentiment analysis?",
        "How has the development of machine learning methods in multi-domain sentiment analysis progressed over the last five years?"
    ]
    # Step 1: Crossref search
    print("Searching Crossref...")
    items = crossref_search(CONFIG["query"], max_results=CONFIG["max_results"],
                            from_year=CONFIG["filters"]["from_pub_date"],
                            until_year=CONFIG["filters"]["until_pub_date"])
    print(f"Crossref returned {len(items)} items (raw).")
    # export .bib (full list)
    bib_path = OUT / "search_results.bib"
    export_bib_from_crossref_items(items, bib_path)
    print("Exported .bib to", bib_path)

    # remove duplicates by DOI
    unique = {}
    dup_count = 0
    for i in items:
        doi = i.get("DOI","").lower()
        if not doi:
            # use title-key fallback
            key = ("title:" + " ".join(i.get("title",[]))).lower()
        else:
            key = doi
        if key in unique:
            dup_count += 1
            continue
        unique[key] = i
    records = list(unique.values())
    print(f"After deduplication: {len(records)} unique records (removed {dup_count}).")

    # Step 2: iterate, check OA & optionally download PDF, extract text
    screening_rows = []
    extracted_rows = []
    total = len(records)
    for rec in tqdm(records, desc="Processing records"):
        meta = brief_metadata_from_crossref(rec)
        doi = meta["doi"]
        unp = None
        pdf_url = None
        is_oa = False
        if doi:
            try:
                unp = call_unpaywall(doi)
            except Exception:
                unp = None
        if unp:
            pdf_url = get_best_pdf_link(unp)
            is_oa = unp.get("is_oa", False)
        pdf_path = None
        text_extracted = ""
        if CONFIG["download_pdfs"] and pdf_url and is_oa:
            fn = doi.replace("/", "_") + ".pdf"
            pdf_path = PDF_DIR / fn
            ok = download_pdf(pdf_url, pdf_path)
            if ok:
                text_extracted = extract_text_from_pdf(pdf_path)
            else:
                pdf_path = None
        # If no pdf text, use Crossref abstract
        if not text_extracted:
            text_extracted = meta.get("abstract","") or ""

        # Screening: rule-based
        rule_decision, rule_reason = rule_based_screen(meta["title"], meta.get("abstract",""))
        # semantic screening
        sim_score, best_rq, sim_decision = semantic_screen(meta["title"], meta.get("abstract",""), RQS)

        # Final decision logic:
        if rule_decision is True:
            final_decision = "Include"
            reason = rule_reason
        elif rule_decision is False:
            final_decision = "Exclude"
            reason = rule_reason
        else:
            # undecided by rules -> use semantic
            if sim_decision:
                final_decision = "Include"
                reason = f"semantic match (score={sim_score:.3f}) to RQ: {best_rq}"
            else:
                final_decision = "Exclude"
                reason = f"semantic non-match (score={sim_score:.3f})"

        screening_rows.append({
            "doi": doi, "title": meta["title"], "authors": meta["authors"],
            "year": meta["year"], "journal": meta["journal"],
            "is_oa": is_oa, "pdf_url": pdf_url or "",
            "screen_decision": final_decision,
            "screen_reason": reason,
            "similarity": sim_score
        })

        if final_decision == "Include":
            # Extract data fields to extracted table (attempt heuristics; manual review recommended)
            # Try to find 'method' keywords in text_extracted
            method_snippet = ""
            lower = text_extracted.lower()
            for kw in ["chunk", "chunking", "semantic", "sliding", "fixed-size", "hierarch"]:
                if kw in lower:
                    i = lower.find(kw)
                    method_snippet = text_extracted[max(0, i-200):i+200].replace("\n"," ")[:800]
                    break
            findings = (lower[:400]).replace("\n"," ")
            extracted_rows.append({
                "doi": doi, "title": meta["title"], "authors": meta["authors"],
                "year": meta["year"], "journal": meta["journal"],
                "method_snippet": method_snippet,
                "key_findings_snippet": findings,
                "fulltext_available": bool(text_extracted.strip()),
                "source_pdf_path": str(pdf_path) if pdf_path else ""
            })

    # Save screening log
    df_screen = pd.DataFrame(screening_rows)
    screen_csv = OUT / "screening_log.csv"
    df_screen.to_csv(screen_csv, index=False)
    print("Saved screening log to", screen_csv)

    # Save extracted data
    df_ext = pd.DataFrame(extracted_rows)
    extracted_csv = OUT / "extracted_data.csv"
    df_ext.to_csv(extracted_csv, index=False)
    print("Saved extracted data to", extracted_csv)

    # PRISMA counts
    counts = {
        "total_records": len(items),
        "duplicates_removed": dup_count,
        "screened_title_abstract": len(records),
        "excluded_title_abstract": int((df_screen["screen_decision"]=="Exclude").sum()),
        "full_text_assessed": int((df_screen["screen_decision"]=="Include").sum()),
        "excluded_full_text": 0,  # requires later manual fulltext exclusions
        "included_final": int((df_screen["screen_decision"]=="Include").sum()),
    }
    prisma_png = OUT / "prisma_flow.png"
    make_prisma(counts, prisma_png)
    print("Saved PRISMA diagram to", prisma_png)

    print("All done. Outputs in:", OUT.absolute())

if __name__ == "__main__":
    main()
