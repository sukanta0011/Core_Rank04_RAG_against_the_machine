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
uv run python3 -m student index --max_chunk_size 1000

# Search
uv run python3 -m student search --query "what is vLLM" --k 10

# Search Dataset
uv run python3 -m student search_dataset --dataset_path "datasets_public/public/AnsweredQuestions/dataset_code_public.json" --k 10 --save_directory "data/output/sources.json"

# Answer
uv run python3 -m student answer --question "what is vLLM?" --k 10

# Answer dataset
uv run python3 -m student answer_dataset --student_search_results_path "data/output/sources.json" --save_directory "data/output/source_answers.json"

# evaluate
uv run python3 -m  student evaluate --student_answer_path "data/output/sources.json" --dataset_path "datasets_public/public/AnsweredQuestions/dataset_code_public.json" --k 10

```
