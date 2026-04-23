import os
import markdown
from xhtml2pdf import pisa

def md_to_pdf(md_file, pdf_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Add basic styling to make the PDF look like a formal document
    styled_html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 14px;
            color: #333333;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 5px;
            font-size: 24px;
        }}
        h2 {{
            color: #34495e;
            font-size: 20px;
            margin-top: 20px;
        }}
        h3 {{
            color: #34495e;
            font-size: 16px;
        }}
        p {{
            margin-bottom: 10px;
        }}
        code {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: "Courier New", Courier, monospace;
            font-size: 12px;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            word-wrap: break-word;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #bdc3c7;
            margin: 20px 0;
        }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    with open(pdf_file, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=result_file)
        
    if pisa_status.err:
        print(f"Error generating PDF {pdf_file}")
    else:
        print(f"Successfully generated {pdf_file}")

def main():
    files = {
        "HLD.md": "HLD Document.pdf",
        "LLD.md": "LLD Document.pdf",
        "Technical_Documentation.md": "Technical Documentation.pdf"
    }
    
    for md_filename, pdf_filename in files.items():
        if os.path.exists(md_filename):
            print(f"Converting {md_filename} to {pdf_filename}...")
            md_to_pdf(md_filename, pdf_filename)
        else:
            print(f"Warning: {md_filename} not found.")

if __name__ == "__main__":
    main()
