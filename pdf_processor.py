# pdf_processor.py

import os
import re
import openai
import torch
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
import pdfplumber

class PDFProcessor:
    """Handles PDF text extraction and QA pair generation using OpenAI API."""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = ' '.join(text.split())  # Remove extra whitespace
        text = ''.join(char for char in text if char.isprintable())  # Remove non-printable characters
        return text.strip()

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a single PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            return self._clean_text(text)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""

    def extract_text_from_pdfs(self, pdf_dir: str) -> List[str]:
        """Extract text from all PDFs in a directory using parallel processing."""
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        with Pool(cpu_count()) as pool:
            all_text = pool.map(self._extract_text_from_pdf, pdf_files)
        return [text for text in all_text if text]

class QAProcessor:
    """Handles generation of Q&A pairs using OpenAI's GPT model."""
    
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature
    
    def generate_qa_pairs(self, texts: List[str], num_pairs: int = 5) -> List[Dict[str, str]]:
        """Generate question-answer pairs using OpenAI's GPT-4."""
        qa_pairs = []
        for text in texts:
            prompt = f"""
            Given the following document text, generate {num_pairs} question-answer pairs:
            - Focus on key information, relationships, and details.
            - Format: "Q: <question> A: <answer>"
            
            Text: {text}
            """
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating Q&A pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            pairs = self._parse_qa_pairs(response['choices'][0]['message']['content'])
            qa_pairs.extend(pairs)
        return qa_pairs
    
    def _parse_qa_pairs(self, response_text: str) -> List[Dict[str, str]]:
        """Parse the raw response into structured Q&A pairs."""
        lines = response_text.split("\n")
        pairs = []
        for line in lines:
            if line.startswith("Q:"):
                question = line.replace("Q:", "").strip()
            elif line.startswith("A:"):
                answer = line.replace("A:", "").strip()
                pairs.append({"question": question, "answer": answer})
        return pairs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PDFs and generate Q&A pairs using AI.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDFs")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API Key")
    args = parser.parse_args()

    processor = PDFProcessor(args.openai_key)
    texts = processor.extract_text_from_pdfs(args.pdf_dir)
    
    qa_processor = QAProcessor()
    qa_pairs = qa_processor.generate_qa_pairs(texts)
    
    for pair in qa_pairs:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}\n")
