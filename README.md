# zjy-nlp-2603

An NLP final project that studies how AI affects different industries by mining a large news corpus. The project builds a notebook-driven pipeline for cleaning web-crawled articles, extracting entities with Gemini, clustering article summaries into topics, classifying sentiment with a fine-tuned transformer, and analyzing cross-industry trends over time.

## Project Goal

The main research question is: how is AI discussed across industries, companies, and technologies, and what does that imply about AI's perceived impact?

The pipeline combines:

- rule-based preprocessing for noisy web-crawled news
- Gemini-based summarization and information extraction
- Qwen embeddings + BERTopic/HDBSCAN clustering
- transformer-based sentiment classification
- downstream exploratory analysis by topic, entity, industry, and month

## Repository Contents

Core notebooks:

- `01_data_cleaning.ipynb`: load raw news data, clean crawl noise, summarize each article
- `02_ner.ipynb`: extract organization, industry, impact, and technology fields with Gemini
- `03_topic_modeling.ipynb`: embed summaries, cluster them, and label clusters
- `04_sentiment_model_training.ipynb`: train sentiment classifier and apply it to clustered summaries
- `05_sentiment_impact_analysis.ipynb`: analyze sentiment and impact patterns
- `06_sentiment_trends_over_time.ipynb`: analyze temporal and industry-level trends

Helper scripts:

- `gemini_utility.py`: threaded Gemini JSON client with retry + cache writing
- `embedding.py`: wrapper for loading Qwen embedding models

Cached artifacts already included:

- `sums.zip`: cached summary responses from Gemini
- `ner.zip`: cached entity extraction responses
- `topics.zip`: cached topic-label responses
- `022_summary_embeddings.npy`: precomputed summary embeddings
- `020_labeled_data.parquet`
- `021_filtered_data.parquet`
- `024_labeled_cluster_data.parquet`
- `040_sentiment.parquet`

## End-to-End Pipeline

### 1. Data Cleaning and Summarization

`01_data_cleaning.ipynb` loads the raw article dataset from either:

- local `news_final_project.parquet`, or
- `https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet`

It then:

- strips empty titles/text
- removes duplicate rows by title
- removes crawl artifacts such as HTML, URLs, boilerplate phrases, and obvious JS/CSS fragments
- saves cleaned text to `010_preprocess.parquet`
- summarizes each article into 1-2 sentences with Gemini
- saves the final summary dataset to `011_summary.parquet`

Notebook notes indicate:

- raw dataset size is about 200K rows
- after cleaning, the checkpoint has 164,151 rows
- a small number of Gemini responses fail and are dropped

### 2. Entity and Impact Extraction

`02_ner.ipynb` reads `011_summary.parquet` and uses Gemini to extract:

- `organization`
- `industry`
- `impact`
- `technology`

The notebook maps impact labels to integers:

- `strong positive -> 2`
- `positive -> 1`
- `neutral -> 0`
- `negative -> -1`
- `strong negative -> -2`

Outputs:

- `020_labeled_data.parquet`: merged extraction results
- `021_filtered_data.parquet`: filtered version after dropping vague/empty entities and overly broad industry values

### 3. Topic Modeling

`03_topic_modeling.ipynb` reads `021_filtered_data.parquet`, embeds article summaries, and clusters them.

Implementation details:

- embedding model loaded via `embedding.py`
- `test_mode=True` uses `Qwen/Qwen3-Embedding-0.6B`
- precomputed embeddings are loaded from `022_summary_embeddings.npy`
- dimensionality reduction: UMAP
- clustering: HDBSCAN
- topic modeling framework: BERTopic

Chosen clustering parameters:

- `n_neighbors=40`
- `n_components=5`
- `min_dist=0.0`
- `min_cluster_size=20`
- `min_samples=1`

Outputs:

- `023_clustered_data.parquet`: summaries with cluster IDs
- `024_labeled_cluster_data.parquet`: clusters further labeled with Gemini-generated topic names

Note: the notebook prints `Number of topics: 1712`, which is a very fine-grained clustering result.

### 4. Sentiment Model Training and Inference

`04_sentiment_model_training.ipynb` fine-tunes a transformer classifier and applies it to the clustered summary dataset.

Actual code behavior:

- base model: `distilroberta-base`
- training dataset used in code: `fhamborg/news_sentiment_newsmtsc`
- output directory: `sentiment_distilroberta_financial_phrasebank/`

This means the notebook markdown and the implementation are not fully aligned: the markdown says Financial PhraseBank, but the executable code currently loads `fhamborg/news_sentiment_newsmtsc`.

The notebook:

- builds a train/validation split
- tokenizes text with `AutoTokenizer`
- fine-tunes with Hugging Face `Trainer`
- evaluates accuracy / precision / recall / F1
- saves the trained model
- runs inference over `024_labeled_cluster_data.parquet`
- writes final predictions to `040_sentiment.parquet`

Final sentiment fields added:

- `sentiment`
- `score`

### 5. Sentiment and Impact Analysis

`05_sentiment_impact_analysis.ipynb` analyzes `040_sentiment.parquet` after excluding outlier cluster `-1`.

Main analyses:

- overall sentiment distribution
- topic-level sentiment score
- organization-level sentiment patterns
- technology-level sentiment patterns
- industry-level sentiment patterns
- disagreement between sentiment and impact

The notebook defines:

- `sentiment_score = positive_share - negative_share`

Some notebook takeaways:

- overall sentiment is mostly positive
- the most positive topic by average impact is Nvidia/AI chips
- the most negative topic by average impact is AI existential risk
- sentiment and impact disagree in about 21.13% of rows

### 6. Trends Over Time

`06_sentiment_trends_over_time.ipynb` uses `040_sentiment.parquet` to visualize:

- monthly aggregate sentiment trends
- topic / industry / technology trend lines
- exposure heuristics for industries
- organizations most discussed in exposed industries
- heuristic mechanisms for how AI may affect them
- heuristic success and risk factors for adoption

Important caveat from the notebook:

- later outlook charts are heuristic summaries inferred from article text, not causal forecasts

## Data Files and Outputs

Typical artifact flow:

1. `news_final_project.parquet`
2. `010_preprocess.parquet`
3. `011_summary.parquet`
4. `020_labeled_data.parquet`
5. `021_filtered_data.parquet`
6. `022_summary_embeddings.npy`
7. `023_clustered_data.parquet`
8. `024_labeled_cluster_data.parquet`
9. `040_sentiment.parquet`

Not all intermediate files are committed. In particular, `011_summary.parquet` and `023_clustered_data.parquet` are generated by notebooks and may need to be rebuilt if missing.

## Environment Setup

This project uses `uv` and requires Python 3.12+.

Install dependencies:

```bash
uv sync
```

Or, if you prefer creating the environment first:

```bash
uv venv
uv sync
```

Open notebooks:

```bash
uv run jupyter lab
```

## External Dependencies

### Gemini API

The Gemini helper reads the API key from:

- `genai_api_key.txt`

Create that file in the project root with your Gemini API key if you want to rerun:

- article summarization
- entity extraction
- topic labeling

If you only want to inspect results, the repository already includes cached zip files for those steps.

### GPU / CUDA

Topic modeling uses:

- `cuml.manifold.UMAP` when CUDA is available
- otherwise `umap.UMAP`

The dependency file also includes:

- `cuml-cu13` on Linux only

So full GPU acceleration is mainly intended for Linux CUDA environments. On macOS, the notebook should fall back to CPU UMAP.

## Reproducing the Pipeline

Recommended order:

1. Run `01_data_cleaning.ipynb`
2. Run `02_ner.ipynb`
3. Run `03_topic_modeling.ipynb`
4. Run `04_sentiment_model_training.ipynb`
5. Run `05_sentiment_impact_analysis.ipynb`
6. Run `06_sentiment_trends_over_time.ipynb`

Practical shortcut:

- keep the provided `sums.zip`, `ner.zip`, `topics.zip`, and `022_summary_embeddings.npy`
- let the notebooks load cached results instead of recomputing all API/model steps

## Key Libraries

- `polars`
- `pandas`
- `pyarrow`
- `sentence-transformers`
- `bertopic`
- `hdbscan`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`
- `google-genai`
- `matplotlib`
- `seaborn`

## Known Caveats

- The project is notebook-first, so there is limited packaging, testing, or CLI automation.
- Several steps rely on cached artifacts instead of fully reproducible scripts.
- Notebook text and code are slightly inconsistent in the sentiment-training section.
- Gemini-driven steps depend on prompt behavior and API responses, so exact results may vary if rerun.
- Some analysis charts in the final notebook are heuristic and should not be treated as causal claims.

## Suggested Improvements

- convert each notebook stage into a parameterized Python module or CLI
- add a single pipeline driver script
- version prompts used for Gemini extraction and labeling
- store evaluation metrics and plots under a dedicated `artifacts/` folder
- add a top-level report summarizing main findings with figures

## Authoring Note

This README was generated by reading the notebooks, helper scripts, dependency metadata, and artifact files present in the repository. It documents the code and data flow as currently implemented, not an idealized project plan.
