from sagemaker.huggingface import HuggingFaceModel,HuggingFacePredictor
import sagemaker

"""
Reusing an existing SageMaker endpoint.

NOTE: SageMaker endpoints are NOT public - they require AWS credentials.
The endpoint URL you see is only accessible with proper AWS authentication.
Make sure your AWS credentials are configured (via ~/.aws/credentials, 
environment variables, or IAM role if running on EC2).
"""


def fetch_documents_from_robust04(n: int = None, index_name: str = "robust04") -> dict[str, str]:
    """Fetch documents from the robust04 pyserini index.
    
    Args:
        n: Number of documents to fetch. If None, fetches all documents.
        index_name: The pyserini prebuilt index name (default: "robust04")
    
    Returns:
        A dictionary mapping docid to document contents.
        Example: {"FT911-1234": "document content...", "FT911-1235": "..."}
    
    Example:
        >>> # Fetch first 10 documents
        >>> docs = fetch_documents_from_robust04(n=10)
        >>> print(f"Fetched {len(docs)} documents")
        >>> 
        >>> # Fetch all documents
        >>> all_docs = fetch_documents_from_robust04()
        >>> print(f"Total documents: {len(all_docs)}")
    """
    from rag.lucene_backend import get_searcher, get_index_reader, fetch_doc_contents
    
    searcher = get_searcher(index_name)
    reader = get_index_reader(index_name)
    
    # Get docids from the index - try multiple methods
    docids = []
    
    # Method 1: Try IndexReader convenience APIs (newer Pyserini)
    if hasattr(reader, "docids"):
        docids = list(reader.docids())
    elif hasattr(reader, "get_docids"):
        docids = list(reader.get_docids())
    else:
        # Method 2: Try to access underlying Lucene reader
        jreader = None
        for attr in ("reader", "_reader", "index_reader", "_index_reader", "lucene_reader", "_lucene_reader"):
            jreader = getattr(reader, attr, None)
            if jreader is not None:
                break
        
        if jreader is not None:
            # Iterate through internal doc IDs
            try:
                max_doc = int(jreader.maxDoc())
            except:
                try:
                    max_doc = int(jreader.numDocs())
                except:
                    max_doc = 0
            
            for i in range(max_doc):
                try:
                    d = jreader.document(i)
                except:
                    try:
                        sf = jreader.storedFields()
                        d = sf.document(i)
                    except:
                        continue
                
                # Extract docid from document
                docid = None
                for k in ("id", "docid", "docno", "DOCNO"):
                    try:
                        v = d.get(k)
                        if isinstance(v, str) and v:
                            docid = v
                            break
                    except:
                        continue
                
                if docid:
                    docids.append(docid)
        else:
            # Method 3: Fallback to searcher iteration
            try:
                num_docs = searcher.num_docs
            except:
                try:
                    num_docs = searcher.get_num_docs()
                except:
                    num_docs = 0
            
            for i in range(num_docs):
                try:
                    d = searcher.doc(i)
                    docid = None
                    for k in ("id", "docid", "docno", "DOCNO"):
                        try:
                            v = d.get(k)
                            if isinstance(v, str) and v:
                                docid = v
                                break
                        except:
                            continue
                    if docid:
                        docids.append(docid)
                except:
                    continue
    
    # Limit to first n if specified
    if n is not None and n > 0:
        docids = docids[:n]
    
    # Fetch contents for each docid
    results = {}
    for docid in docids:
        content = fetch_doc_contents(searcher, docid)
        if content:  # Only include non-empty documents
            results[docid] = content
    
    return results

endpoint_name = "huggingface-pytorch-inference-2026-01-09-08-33-10-953"
predictor = HuggingFacePredictor(endpoint_name=endpoint_name)

# role = 'arn:aws:iam::258974340175:role/SageMakerExecutionRole'
# hub = {
#     "HF_MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2",  # 8192 tokens, no trust_remote_code
#     "HF_TASK": "feature-extraction"
# }
# huggingface_model = HuggingFaceModel(
#     env=hub,
#     role=role,
#     transformers_version="4.37",
#     pytorch_version="2.1",
#     py_version="py310"
# )
# predictor = huggingface_model.deploy(
#     initial_instance_count=1,
#     instance_type="ml.g4dn.xlarge"
# )

docs = fetch_documents_from_robust04(n=10)

# Load tokenizer for proper truncation
from transformers import AutoTokenizer
import re

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def extract_text_paragraphs(doc: str) -> str:
    """Extract paragraph content from within <text> tags of robust04 documents.
    
    Args:
        doc: Raw document with XML-like tags (<date>, <headline>, <text>, <p>, etc.)
    
    Returns:
        Clean text with just the paragraph contents joined together.
    """
    # Extract content within <text>...</text>
    text_match = re.search(r'<\s*text\s*>(.*?)<\s*/\s*text\s*>', doc, re.IGNORECASE | re.DOTALL)
    if not text_match:
        # Fallback: try to extract <p> tags from the whole document
        text_content = doc
    else:
        text_content = text_match.group(1)
    
    # Extract all <p>...</p> contents
    paragraphs = re.findall(r'<\s*p\s*>(.*?)<\s*/\s*p\s*>', text_content, re.IGNORECASE | re.DOTALL)
    
    if not paragraphs:
        # Fallback: strip all tags and return cleaned text
        clean = re.sub(r'<[^>]+>', ' ', doc)
        return ' '.join(clean.split())
    
    # Join paragraphs with newlines and clean up whitespace
    clean_paragraphs = []
    for p in paragraphs:
        # Remove any remaining tags and normalize whitespace
        clean_p = re.sub(r'<[^>]+>', ' ', p)
        clean_p = ' '.join(clean_p.split())
        if clean_p:
            clean_paragraphs.append(clean_p)
    
    return '\n\n'.join(clean_paragraphs)

import numpy as np

CHUNK_SIZE = 450  # Leave room for special tokens (512 - 62 buffer)
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks based on token count.
    
    Args:
        text: The text to chunk
        chunk_size: Max tokens per chunk
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Move start forward by (chunk_size - overlap)
        start += chunk_size - overlap
        
        # Avoid tiny final chunks
        if len(tokens) - start < overlap:
            break
    
    return chunks

def embed_with_chunking(text: str, strategy: str = "mean") -> np.ndarray:
    """Embed a long document using chunking.
    
    Args:
        text: The document text
        strategy: "mean" to average chunk embeddings, "first" to use only first chunk
    
    Returns:
        Document embedding as numpy array
    """
    chunks = chunk_text(text)
    print(f"  Split into {len(chunks)} chunks")
    
    def extract_embedding(result):
        """Extract flat embedding vector from API response."""
        emb = result[0]
        # Unwrap nested lists
        while isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
            emb = emb[0]
        # If still a nested list (token embeddings), mean pool them
        if isinstance(emb, list) and isinstance(emb[0], list):
            emb = np.mean(emb, axis=0)
        return np.array(emb)
    
    if strategy == "first":
        result = predictor.predict({"inputs": [chunks[0]]})
        return extract_embedding(result)
    
    # Embed chunks sequentially (parallelism handled at document level)
    chunk_embeddings = []
    for chunk in chunks:
        result = predictor.predict({"inputs": [chunk]})
        chunk_embeddings.append(extract_embedding(result))
    
    # Mean pool all chunk embeddings
    return np.mean(chunk_embeddings, axis=0)

# Process and predict with threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_THREADS = 8  # Number of parallel document processors

def process_document(docid: str, raw_doc: str) -> tuple[str, np.ndarray, int, int, float]:
    """Process a single document and return its embedding."""
    doc_start = time.time()
    
    clean_text = extract_text_paragraphs(raw_doc)
    num_tokens = len(tokenizer.encode(clean_text))
    chunks = chunk_text(clean_text)
    doc_embedding = embed_with_chunking(clean_text, strategy="mean")
    
    doc_time = time.time() - doc_start
    return docid, doc_embedding, num_tokens, len(chunks), doc_time

print(f'\nPredicting with chunking (using {NUM_THREADS} parallel workers):')
total_start = time.time()
results = {}
doc_times = []

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    # Submit all documents in parallel
    futures = {
        executor.submit(process_document, docid, raw_doc): docid 
        for docid, raw_doc in docs.items()
    }
    
    # Process results as they complete
    for future in as_completed(futures):
        docid, doc_embedding, num_tokens, num_chunks, doc_time = future.result()
        results[docid] = doc_embedding
        doc_times.append(doc_time)
        
        print(f"[{docid}] {num_tokens} tokens, {num_chunks} chunks -> shape {doc_embedding.shape} in {doc_time:.2f}s")

total_time = time.time() - total_start
print(f"\n{'='*80}")
print(f"TIMING SUMMARY (Threaded)")
print(f"{'='*80}")
print(f"Total documents: {len(docs)}")
print(f"Threads used: {NUM_THREADS}")
print(f"Total time: {total_time:.2f}s")
print(f"Avg time per doc (wall): {total_time/len(docs):.2f}s")
print(f"Avg time per doc (CPU): {sum(doc_times)/len(doc_times):.2f}s")
print(f"Min/Max: {min(doc_times):.2f}s / {max(doc_times):.2f}s")
print(f"Throughput: {len(docs)/total_time:.2f} docs/sec")
print(f"Speedup vs sequential: ~{sum(doc_times)/total_time:.1f}x")