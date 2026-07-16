# mech-interp

A hands-on introduction to mechanistic interpretability: the attempt to reverse-engineer what's actually happening inside a trained neural network, rather than just measuring its inputs and outputs. Instead of treating a language model as a black box, mech interp opens it up and asks concrete questions like "which attention head is copying this token?" or "which direction in the residual stream corresponds to this token's logit?", and answers them by running real experiments on a real model (GPT-2, in this repo) rather than reasoning from theory alone.

This repo is a self-paced track: a short library warm-up, then a series of notebooks that build up the core toolkit (TransformerLens, activation caching, hooks) and use it to find and explain real circuits inside GPT-2, such as induction heads.

## Repo structure

```
books/           reference textbooks and papers, read on demand, not cover to cover
00_libraries/    interactive tutorials for the libraries everything else depends on
01_notebooks/    worked mech interp notebooks, fully filled in, read and run them
02_exercises/    the same notebooks with key lines blanked out, fill them in yourself
env.yml          conda environment definition
```

### `00_libraries/`

Four "type it yourself" tutorials, in order: `00_numpy`, `01_torch`, `02_pandas`, `03_transformer_lens`. Each one is reference code in markdown followed by a blank code cell; type the example yourself rather than copy-pasting, and finish with a self-checked exercise. These aren't generic library tutorials: they specifically cover the operations that show up later in `02_exercises/` (flattening a `[layer, head]` grid to find the top-scoring head, extracting a diagonal from an attention pattern, editing an activation in place inside a hook).

### `01_notebooks/`

The actual mech interp content, meant to be read and run in order:

1. `00_transformerlens_tour.ipynb`: TransformerLens mechanics (loading a model, `run_with_cache`, hooks), no interpretability theory yet.
2. `01_transformer_primer.ipynb`: the residual stream, attention, and MLPs, explained using the tools from step 1.
3. `02_transformerlens_basics.ipynb`: more practice with caching and reading activations.
4. `03_induction_heads.ipynb`: finding the attention heads responsible for in-context copying.
5. `04_logit_lens.ipynb`: reading out predictions from intermediate layers, and direct logit attribution per head.
6. `05_activation_patching.ipynb`: causal interventions, patching one run's activations into another to test which component matters.

### `02_exercises/`

The same three later notebooks (`03`, `04`, `05`) with the key computation in each blanked out. Do these after the matching `01_notebooks/` notebook, once you understand what the code should do, not before.

## Setup

```
conda env create -f env.yml
conda activate mech-interp
jupyter lab
```

GPU is used automatically if available (`torch.cuda.is_available()`), but everything also runs on CPU.

## Suggested order

1. Work through `00_libraries/` in order, actually typing the code rather than skimming it.
2. Read and run each `01_notebooks/` notebook in order.
3. After each of `03`, `04`, `05`, immediately do the matching `02_exercises/` notebook while it's still fresh.
4. Use `books/` as a reference, not a syllabus: when a notebook introduces a concept you can't yet turn into code, look it up, read just enough to unblock yourself, and get back to the notebook.
