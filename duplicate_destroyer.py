import bibtexparser

def deduplicate_bib(input_file, output_file, dedup_field="doi"):
    # Load bib file
    with open(input_file, "r", encoding="utf-8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    total_entries = len(bib_database.entries)

    # Deduplicate by the chosen field
    unique_entries = {}
    for entry in bib_database.entries:
        key = entry.get(dedup_field)
        if key and key not in unique_entries:
            unique_entries[key] = entry

    deduped_entries = list(unique_entries.values())
    duplicates_removed = total_entries - len(deduped_entries)

    # Save deduplicated bib file
    bib_database.entries = deduped_entries
    with open(output_file, "w", encoding="utf-8") as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

    # Log statement
    print(
        f"Duplicate Log: Retrieved {total_entries} articles, removed {duplicates_removed} duplicates. "
        f"{len(deduped_entries)} unique articles will proceed to the screening phase."
    )

# main
deduplicate_bib("compiled.bib", "output.bib", dedup_field="doi")

# DEBUGGING
with open("compiled.bib", "r", encoding="utf-8") as bibtex_file:
    bib_str = bibtex_file.read()

print("Raw .bib length:", len(bib_str.split("@")) - 1, "entries detected by string count")

from bibtexparser.bparser import BibTexParser
parser = BibTexParser(common_strings=True)
bib_database = bibtexparser.loads(bib_str, parser=parser)

print("bibtexparser loaded:", len(bib_database.entries), "entries")
