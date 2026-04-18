# RAG against the machine

- Ingest (bm25, chromadb)
• Read and process all files from the vLLM repository provide in the attachments
• Implement intelligent chunking for Python code and Markdown documentation
• Create a searchable index using TF-IDF or BM25
• Store the index for fast retrieval (maximum 5 minutes indexing time)(chromadb)
• Python code chunking
• Text chunking
The maximum chunk size is 2000 characters and it has to be
configurable through a CLI argument.
• Repository: Index all the files you judge useful in the repository
For each query, your system must retrieve relevant chunks of the repository and generate
an evidence-based response in the same form as the output


- search()
• TF-IDF
• BM25
• Implement semantic search over the indexed knowledge base
• Return top-k most relevant code snippets for any query
• Each result must include: file_path, first_character_index, last_character_index
• Support batch processing of multiple questions from JSON datasets
• Achieve at least 55% recall@5 on docs questions and 45% on code questions

- Answer (dspy, langchain, transformer)
• Use Qwen/Qwen3-0.6B model to generate natural language answers (transformer)
• Pass retrieved context to the LLM within token limits
• Generate answers based on the retrieved code and documentation
• Output structured JSON following the provided pydantic models
• Answer questions in maximum 2 seconds per question

- Evaluate
• Implement recall@k metric to measure retrieval quality
• Compare retrieved sources against ground truth annotations
• Calculate overlap between retrieved and correct sources (minimum 5% overlap
counts as found)
• Provide detailed performance metrics
• Indexing time: 5 minutes maximum
• Cold start latency: 60 seconds maximum (first retrieval after system startup,
including model loading)
• Warm retrieval throughput: 90 seconds maximum for 1000 questions (after cold
start)
• Recall@5: 55% on docs questions and 45% on code


-CLI(fire, tqdm)
• Provide a CLI using Python Fire with these commands:
◦ index: Index the repository
◦ search: Search for a single query
◦ search_dataset: Process multiple questions and output search results
◦ answer: Answer a single question with context
◦ answer_dataset: Generate answers from search results
◦ evaluate: Evaluate search results against ground truth
• Include progress bars for long-running operations
• Handle errors gracefully with clear messages

pydantic
MinimalSource, UnansweredQuestion, AnsweredQuestion, RagDataset
MinimalSearchResults, MinimalAnswer,
StudentSearchResults, StudentSearchResultsAndAnswer


## CLI commands:
```bash
# Indexing
uv run python3 -m src index --max_chunk_size 1000

# Search
uv run python3 -m src search --query "what is vLLM" --k 10

# Search Dataset
uv run python3 -m src search_dataset --dataset_path "datasets_public/public/AnsweredQuestions/dataset_code_public.json" --k 10 --save_directory "data/output/sources.json"

# Answer
uv run python3 -m src answer --question "what is vLLM?" --k 10

# Answer dataset
uv run python3 -m src answer_dataset --student_search_results_path "data/output/sources.json" --save_directory "data/output/source_answers.json"

# evaluate
uv run python3 -m  src evaluate --student_answer_path "data/output/sources.json" --dataset_path "datasets_public/public/AnsweredQuestions/dataset_code_public.json" --k 10

# Moulinette evaluation
uv run moulinette evaluate_student_search_results data/output/sources.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json  --k 10 --max_context_length 2000 --threshold 0.80

```

# Results:

Model Name,Size (MB),Strengths,Why for you?

all-MiniLM-L6-v2,~80,General English,"Your current ""baseline."""

multi-qa-mpnet-base-dot-v1,~420,Q&A specifically,Trained on StackOverflow/Reddit Q&A.

bge-small-en-v1.5,~130,Massive Retrieval Data,High accuracy in technical docs.


nomic-embed-text-v1,~270,Long Context (8k),Perfect for your 2000-character chunks.

chunk size 2000, overlap 50
docs: 100 Questions
BM25-lexical (~0.25s)
📈 Recall@1: 0.600 (60.0%)
📈 Recall@3: 0.790 (79.0%)
📈 Recall@5: 0.840 (84.0%)
📈 Recall@10: 0.930 (93.0%)

MiniLM_L6_v2-Semantic (~1.2s)
📈 Recall@1: 0.290 (29.0%)
📈 Recall@3: 0.420 (42.0%)
📈 Recall@5: 0.490 (49.0%)
📈 Recall@10: 0.550 (55.0%)

Hybrid-rrf (~1.35s)
📈 Recall@1: 0.430 (43.0%)
📈 Recall@3: 0.670 (67.0%)
📈 Recall@5: 0.770 (77.0%)
📈 Recall@10: 0.860 (86.0%)

Hybrid-cross_validation (~10s)
📈 Recall@1: 0.610 (61.0%)
📈 Recall@3: 0.750 (75.0%)
📈 Recall@5: 0.780 (78.0%)
📈 Recall@10: 0.870 (87.0%)


chunk size 2000, overlap 50
code: 100 Questions
BM25-lexical (~0.25s)
📈 Recall@1: 0.290 (29.0%)
📈 Recall@3: 0.480 (48.0%)
📈 Recall@5: 0.540 (54.0%)
📈 Recall@10: 0.570 (57.0%)

MiniLM_L6_v2-Semantic (~1.2s)
📈 Recall@1: 0.190 (19.0%)
📈 Recall@3: 0.280 (28.0%)
📈 Recall@5: 0.370 (37.0%)
📈 Recall@10: 0.480 (48.0%)

Hybrid-rrf (~1.3s)
📈 Recall@1: 0.270 (27.0%)
📈 Recall@3: 0.440 (44.0%)
📈 Recall@5: 0.530 (53.0%)
📈 Recall@10: 0.670 (67.0%)

Hybrid-cross_validation (~10s)
📈 Recall@1: 0.380 (38.0%)
📈 Recall@3: 0.570 (57.0%)
📈 Recall@5: 0.610 (61.0%)
📈 Recall@10: 0.670 (67.0%)

# Docker-Setup:
Run these commands one by one:

Update your package list:
sudo apt-get update

Install the official Docker script:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

Add your user to the "docker" group:
By default, Docker needs sudo. Adding yourself to the group lets you run commands like a pro without typing your password every time.
sudo usermod -aG docker $USER


Run these to set up the NVIDIA repository:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Install the toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

Restart the Docker service to apply changes:
sudo systemctl restart docker

Run this command to verify the connection:
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi