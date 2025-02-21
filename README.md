# PDF AI Processor

## Overview
This project automates **PDF text extraction** and **Q&A generation** using **OpenAI's GPT-4**. It allows you to extract text from PDFs and generate structured question-answer pairs for study materials, legal document processing, or business automation.

## Features
✅ Extracts clean text from PDFs using **PyPDF2** and **pdfplumber**  
✅ Generates **intelligent Q&A pairs** from document content using **GPT-4**  
✅ Fine-tunes a smaller LLM (**Phi-2**) for domain-specific tasks  
✅ Supports **parallel processing** for speed optimization  
✅ Simple **command-line interface (CLI)** for easy execution  

## Installation
Ensure you have **Python 3.8+** installed, then install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
To extract text from PDFs and generate Q&A pairs, run:
```sh
python pdf_processor.py --pdf_dir "path/to/pdfs" --openai_key "your_api_key"
```

## Example Output
```
Q: What is the main topic of the document?
A: The document discusses AI-powered document processing.

Q: What technology is used for text extraction?
A: PyPDF2 and pdfplumber are used for extracting text from PDFs.
```

## Future Improvements
- Add **support for different file types** (Word, TXT, etc.)  
- Implement **better error handling and logging**  
- Fine-tune **domain-specific models** for better accuracy  

## License
This project is open-source under the **MIT License**.

---

🚀 **Enjoy automating document analysis with AI!**

