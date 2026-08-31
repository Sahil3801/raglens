from pathlib import Path
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from pypdf import PdfReader
import json

parser = argparse.ArgumentParser(description="Create synthetic PDFs for runtime smoke verification, not a quality benchmark.")
parser.add_argument("--output", type=Path, required=True)
root = parser.parse_args().output.resolve()
root.mkdir(parents=True, exist_ok=True)
stations = [("aurora", "AMBER-731", 17), ("borealis", "COBALT-482", 29)]
manifest = []
for station, code, interval in stations:
    path = root / f"raglens_runtime_20260831_{station}.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    for page in range(1, 37):
        heading = f"{station.title()} Station - Runtime Verification - Page {page}"
        if page == 1:
            body = (
                f"This is synthetic test material for the RagLens application. "
                f"The maintenance access code for {station.title()} station is {code}. "
                f"The inspection interval for {station.title()} station is {interval} days. "
                "These values are fictional identifiers for software verification, not real credentials. "
                "The station handbook contains no personal records, salary details, planetary facts, or weather forecasts."
            )
        else:
            body = (
                f"Procedure {page:02d} covers archive shelf {page:02d} at {station.title()} station. "
                "Before handling a storage box, check its paper label and record the shelf number. "
                "Inspect the box for dust and confirm that its cover is closed. "
                "Keep the reading desk clear and return folders to their original order. "
                "Use the inventory ledger to record completed housekeeping work. "
                "This page describes archive housekeeping only and does not specify the maintenance access code or inspection interval."
            )
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(48, 790, heading)
        pdf.setFont("Helvetica", 11)
        y = 755
        for line in simpleSplit(body, "Helvetica", 11, 490):
            pdf.drawString(48, y, line)
            y -= 17
        pdf.showPage()
    pdf.save()
    reader = PdfReader(path)
    manifest.append({"filename": path.name, "pages": len(reader.pages), "code": code,
                     "interval_days": interval, "extracted_characters": sum(len(p.extract_text()) for p in reader.pages)})
    assert code in reader.pages[0].extract_text()
print(json.dumps(manifest, indent=2))
