import json
from datetime import datetime
from extractor import extract_sections
from relevance_ranker import rank_sections
import os

def main():
    input_dir = '/app/input'
    output_dir = '/app/output'
    
    persona = "PhD Researcher in Computational Biology"
    job_to_be_done = "Prepare a comprehensive literature review focusing on methodologies, datasets, and benchmarks"

    for pdf_file in os.listdir(input_dir):
        sections = extract_sections(os.path.join(input_dir, pdf_file))
        ranked_sections = rank_sections(sections, persona, job_to_be_done)
        
        result = {
            "metadata": {
                "input_documents": pdf_file,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": pdf_file,
                    "page_number": sec["page_number"],
                    "section_title": sec["section_title"],
                    "importance_rank": sec["importance_rank"],
                    "refined_text": sec["content"][:1000]  # Keep short to ensure relevance
                }
                for sec in ranked_sections[:10]  # Select top 10 sections
            ]
        }
        
        output_filename = os.path.splitext(pdf_file)[0] + '.json'
        with open(os.path.join(output_dir, output_filename), 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
