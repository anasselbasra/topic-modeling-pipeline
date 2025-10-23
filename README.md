# Topic Modeling Pipeline

![Annotated cluster visualization](assets/Topic%20modeling%20annotated%20clusters.png)

## Description
This repository contains a modular pipeline for discovering and labelling topics from large collections of short texts.  The workflow combines modern sentence embeddings, density-based clustering, and large language model (LLM) assisted annotation to move from raw documents to labelled, visualisable themes.  It is organised around two notebooks and a small collection of reusable utilities so that you can adapt the process to your own datasets.

### Repository structure
- `1_semantic_stratification_&_topic_inference_from_high-dimensional_embeddings.ipynb` – end-to-end notebook that loads data, creates sentence embeddings, reduces dimensionality with UMAP, discovers clusters with HDBSCAN, and exports intermediate artefacts (embeddings, projections, probabilities).
- `2_annotation_with_llm.ipynb` – placeholder notebook prepared for driving automated topic annotation with LLMs on top of the previously inferred clusters.
- `utils/` – helper modules used by the notebooks:
  - `helper_functions.py` – text cleaning, sampling strategies for LLM prompts, OpenAI annotation helpers, cluster-centre utilities, and hierarchical clustering logic.
  - `helper_functions_visualization.py` – Plotly-driven cluster visualisations (interactive scatter, similarity heatmap, dendrogram) for inspecting the outputs.
  - `helper_functions_documentation_guide.md` – prose documentation explaining the intent and design of the helper utilities.
- `data/` – sample parquet datasets, including an annotated split (`annotated_final_data/`) to experiment with and a raw snapshot (`df_AI_Trends_Linkdin_juin_2025.parquet`).
- `assets/` – static media such as the introductory cluster map used in this README.
- `visualization.html` – generated Plotly dashboard showcasing clustered data; refreshed whenever `visualize_clusters` is executed.

## Installation
1. Ensure you have Python 3.10+ available and (optionally) create an isolated environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
2. Install the Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Optional) If you intend to call the OpenAI API for topic labelling, create an environment variable `OPENAI_API_KEY` or supply the key explicitly when invoking the helper.

## Usage
1. Prepare your data by placing a parquet file containing a text column (named `text` in the default notebook) inside the `data/` directory.
2. Open and execute `1_semantic_stratification_&_topic_inference_from_high-dimensional_embeddings.ipynb` in Jupyter Lab or VS Code:
   - Adjust sampling parameters (e.g., `sample_size`) and embedding model (`SentenceTransformer`) as required.
   - Run the notebook to generate embeddings, 2D UMAP projections, clustering labels, and per-sample probabilities.
   - Save intermediate outputs or reuse the in-memory dataframe for downstream steps.
3. (Optional) Launch `2_annotation_with_llm.ipynb` to orchestrate LLM-assisted topic annotation.  The notebook is currently a scaffold—consult `utils/helper_functions.py` for ready-to-use `call_openai_api`, `select_examples`, and hierarchical clustering helpers when building your workflow.
4. To explore results interactively, import `utils/helper_functions_visualization.py` and call `visualize_clusters` from within a notebook.  This creates `visualization.html`, which you can open in a browser for a Plotly-based overview of topics, annotations, and cluster centres.
5. Extend the analysis by using additional utilities:
   - `similarity_matrix`, `plot_heatmap_inter_cluster_cosine_similarity`, and `plot_dendrogram_with_labels` to examine relationships between clusters.
   - `hirarchical_clustering` to merge highly similar clusters and iteratively refine the taxonomy.

## License
No explicit license is provided.  Please contact the repository owner before reusing the code or data in other projects.
