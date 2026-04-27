# The Deleuzian Representation Hypothesis

Code associated with the [article](https://arxiv.org/abs/2512.19734), ICLR 2026.

> We propose an alternative to sparse autoencoders (SAEs) as a simple and effective unsupervised method for extracting interpretable concepts from neural networks. The core idea is to cluster differences in activations, which we formally justify within a discriminant analysis framework. To enhance the diversity of extracted concepts, we refine the approach by weighting the clustering using the skewness of activations. The method aligns with Deleuze's modern view of concepts as differences. We evaluate the approach across five models and three modalities (vision, language, and audio), measuring concept quality, diversity, and consistency. Our results show that the proposed method achieves concept quality surpassing prior unsupervised SAE variants while approaching supervised baselines, and that the extracted concepts enable steering of a model’s inner representations, demonstrating their causal influence on downstream behavior.

## Installation 
On Linux, using uv, create an environnment from pyproject.toml. 

```bash
uv sync
source .venv/bin/activate
```

## Example Usage

To extract concepts from precomputed activation matrices : 

```python
from extraction import memsafediffs
# Learn concepts from neural activations
acts = torch.load('prerecorded_activations.pt')
deleuzian_proj = memsafediffs(acts, n_dims=1000)
# Apply to new activations
new_acts = torch.load('any_activations.pt')
deleuzian_concepts = deleuzian_proj(new_acts)
```

Utilities:
- `wrappedmodels.py` to hook a model
- `record_acts.py` for activations recording from a model
- `probes.py` to measure probe loss (concept quality, with respect to dataset labels)
- `mppc.py` to measure concept consistency with MPPC
