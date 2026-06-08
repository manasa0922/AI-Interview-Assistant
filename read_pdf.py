from pypdf import PdfReader

pdf_path = "docs/MachineLearning.pdf"

reader = PdfReader("C:\My Projects\AI-Interview-Assistant\docs\MachineLearning.pdf")

print(f"Total Pages: {len(reader.pages)}")

for page in reader.pages:
    text = page.extract_text()
    print(text)