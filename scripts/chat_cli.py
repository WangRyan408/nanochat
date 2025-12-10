"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli -i mid

With RAG (Retrieval Augmented Generation):
python -m scripts.chat_cli --rag --faiss-path medical_index.faiss --metadata-path medical_metadata.pkl

Batch processing mode:
python -m scripts.chat_cli --rag --prompts-file rag/queries.json --output-file results.json --max-prompts 100
"""
import argparse
import json
import pickle
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
# RAG arguments
parser.add_argument('--rag', action='store_true', help='Enable RAG (Retrieval Augmented Generation)')
parser.add_argument('--faiss-path', type=str, default='medical_index.faiss', help='Path to FAISS index file')
parser.add_argument('--metadata-path', type=str, default='medical_metadata.pkl', help='Path to metadata pickle file')
parser.add_argument('--rag-k', type=int, default=5, help='Number of documents to retrieve for RAG context')
# Batch processing arguments
parser.add_argument('--prompts-file', type=str, default='', help='JSON file with prompts for batch processing (expects array with "query" field)')
parser.add_argument('--output-file', type=str, default='batch_results.json', help='Output JSON file for batch results')
parser.add_argument('--max-prompts', type=int, default=None, help='Maximum number of prompts to process in batch mode')
parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum tokens to generate per response')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

# RAG Retriever class
class RAGRetriever:
    """RAG retriever using FAISS index and SentenceTransformer embeddings."""

    def __init__(self, faiss_path: str, metadata_path: str, k: int = 5):
        print("Loading RAG components...")
        
        # Load FAISS index
        print(f"  Loading FAISS index from {faiss_path}...")
        self.index = faiss.read_index(faiss_path)
        print(f"  FAISS index loaded with {self.index.ntotal} vectors")
        
        # Load metadata
        print(f"  Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.texts = self.metadata['texts']
        self.original_data = self.metadata.get('original_data', [])
        print(f"  Metadata loaded with {len(self.texts)} entries")
        
        # Load embedding model (same as used for indexing)
        print("  Loading SentenceTransformer embedding model...")
        self.embed_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
        print("  RAG components ready!")
        
        self.k = k

    def retrieve(self, query: str, k: int = None) -> list:
        """Retrieve the top-k most similar documents for a query."""
        k = k or self.k
        
        # Embed the query (normalize to match index)
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k)
        
        # Gather results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            result = {
                'rank': i + 1,
                'score': float(score),
                'text': self.texts[idx],
            }
            if idx < len(self.original_data):
                result['data'] = self.original_data[idx]
            results.append(result)
        
        return results

    def format_context(self, results: list) -> str:
        """Format retrieved results into a context string for the LLM."""
        if not results:
            return ""
        
        context_parts = ["Here is relevant medical data from similar cases:\n"]
        for r in results:
            context_parts.append(f"[Case {r['rank']} (similarity: {r['score']:.3f})]\n{r['text']}\n")
        
        context_parts.append("\nUse the above information to help answer the user's question.")
        return "\n".join(context_parts)

# Initialize RAG retriever if enabled
rag_retriever = None
if args.rag:
    rag_retriever = RAGRetriever(
        faiss_path=args.faiss_path,
        metadata_path=args.metadata_path,
        k=args.rag_k
    )

# ============================================================================
# BATCH PROCESSING MODE
# ============================================================================
if args.prompts_file:
    print(f"\nBatch Processing Mode")
    print("-" * 50)
    
    # Load prompts from JSON file
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
    
    # Limit number of prompts if specified
    if args.max_prompts is not None:
        prompts_data = prompts_data[:args.max_prompts]
    
    print(f"Loaded {len(prompts_data)} prompts from {args.prompts_file}")
    if args.rag:
        print(f"RAG: enabled (k={args.rag_k})")
    print("-" * 50)
    
    results = []
    
    for i, item in enumerate(prompts_data):
        # Extract the query text
        if isinstance(item, dict):
            user_input = item.get('query', item.get('prompt', str(item)))
            patient_id = item.get('patient_id', None)
            ground_truth = item.get('ground_truth', None)
            features = item.get('features', None)
        else:
            user_input = str(item)
            patient_id = None
            ground_truth = None
            features = None
        
        print(f"\n[{i+1}/{len(prompts_data)}] Processing...", end="", flush=True)
        
        # RAG retrieval if enabled
        augmented_input = user_input
        if rag_retriever is not None:
            rag_results = rag_retriever.retrieve(user_input)
            rag_context = rag_retriever.format_context(rag_results)
            if rag_context:
                augmented_input = f"{rag_context}\n\n---\n\nUser question: {user_input}"
        
        # Build conversation tokens (fresh for each prompt)
        conversation_tokens = [bos]
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(augmented_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        
        # Generate response
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }
        response_tokens = []
        with autocast_ctx:
            for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                token = token_column[0]
                response_tokens.append(token)
        
        # Ensure assistant_end token is present (may be missing if max_tokens was hit)
        if response_tokens and response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        
        # Decode response
        response_text = tokenizer.decode(response_tokens)
        print(f" done ({len(response_tokens)} tokens)")
        
        # Build result entry
        result_entry = {
            "prompt": user_input,
            "response": response_text,
        }
        if patient_id is not None:
            result_entry["patient_id"] = patient_id
        if ground_truth is not None:
            result_entry["ground_truth"] = ground_truth
        if features is not None:
            result_entry["features"] = features
        
        results.append(result_entry)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Processed: {len(results)} prompts")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*50}")
    
    # Exit after batch processing
    exit(0)

# ============================================================================
# INTERACTIVE MODE
# ============================================================================
print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
if args.rag:
    print(f"RAG: enabled (k={args.rag_k})")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # RAG retrieval: augment user input with retrieved context
    augmented_input = user_input
    if rag_retriever is not None:
        results = rag_retriever.retrieve(user_input)
        rag_context = rag_retriever.format_context(results)
        if rag_context:
            augmented_input = f"{rag_context}\n\n---\n\nUser question: {user_input}"
            print(f"[RAG] Retrieved {len(results)} relevant documents")

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(augmented_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
