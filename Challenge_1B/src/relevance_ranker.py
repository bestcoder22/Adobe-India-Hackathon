from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_sections(sections, persona, job):
    query_embedding = model.encode(f"{persona}. {job}")
    
    section_embeddings = model.encode([sec['section_title'] + " " + sec['content'][:500] for sec in sections])
    similarities = cosine_similarity([query_embedding], section_embeddings).flatten()
    
    ranked_sections = sorted([
        {**section, "importance_rank": rank + 1, "similarity": sim} 
        for rank, (section, sim) in enumerate(sorted(zip(sections, similarities), key=lambda x: x[1], reverse=True))
    ], key=lambda x: x['importance_rank'])

    return ranked_sections
