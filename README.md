# The Deleuzian Representation Hypothesis

Code associated with the article. 

> We propose an alternative to sparse autoencoders (SAEs) as a simple and effective unsupervised method for extracting interpretable concepts from neural networks. The core idea is to cluster differences in activations, which we formally justify within a discriminant analysis framework. To enhance the diversity of extracted concepts, we refine the approach by weighting the clustering using the skewness of activations. The method aligns with Deleuze's modern view of concepts as differences. We evaluate the approach across five models and three modalities (vision, language, and audio), measuring concept quality, diversity, and consistency. Our results show that the proposed method achieves concept quality surpassing prior unsupervised SAE variants while approaching supervised baselines, and that the extracted concepts enable steering of a model’s inner representations, demonstrating their causal influence on downstream behavior.

## Installation 
On Linux, using uv, create an environnment from pyproject.toml. 

```bash
uv sync
source .venv/bin/activate
```

## Usage

To use probe loss evaluation to assess whether extracted concepts retain the desired attributes : 

```bash
python eval_features.py --model ClipVision --method large_diffs --dataset ImNet
```

