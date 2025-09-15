import fitz  # PyMuPDF

def highlight_text_in_pdf(pdf_path, highlights, output_path="highlighted.pdf"):
    doc = fitz.open(pdf_path)
    for item in highlights:
        page_number = item["page"]
        text = item["text"]
        try:
            page = doc[page_number]
            text_instances = page.search_for(text)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
        except Exception:
            pass
    doc.save(output_path)
    return output_path
