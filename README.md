# Data availability

The ACS benchmark of minimal pairs of code-switching sentences is available on huggingface: [https://huggingface.co/datasets/igorsterner/acs-benchmark](https://huggingface.co/datasets/igorsterner/acs-benchmark)

We have also make available a large corpus of code-switching: [https://huggingface.co/datasets/igorsterner/acs-corpus](https://huggingface.co/datasets/igorsterner/acs-corpus). The identifiers from the benchmark match up to entries in this corpus. So look here if you want automatic translations, parse trees, alignments or full source tweets for the observed data in the benchmark.


# Generate minimal pairs of code-switching

## Environment

Make sure you have python installed (we used version 3.11.8) and the required dependencies.

```
conda create -n myenv python=3.11.8
conda activate myenv
pip install -r requirements.txt
```

Clone this repository and make sure the python path is set appropriately.

```
git clone https://github.com/igorsterner/acs
cd acs
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Input format

As input, we used raw twitter text. We assume it is provided in JSON-line format, with each line including a "text" field (see e.g. `data/demonstration_example/de-en.jsonl`).

## Tools

Lots of tools are required for preprocessing. Run them all with in the following style (note that English is always the second language in our setup, as the token-based language identification model assumes that)

```
python acs/minimal_pairs/processing.py --lang1 de --lang2 en
```

This creates a cached file of the processing results (by default `data/cache/processed_data.pkl`).


## Minimal pair generation

Now you can use the result in order to generate minimal pairs. Run the following to randomly generate one possible minimal pair for each provided CS sentence.

```
python acs/minimal_pairs/minimal_pairs.py
```

For the provided example, the output is either:

```
> @USER And I said maybe etwas leiser singen, sonst ruf ich die Polizei
> @USER And I said maybe a little leiser singen, sonst ruf ich die Polizei
```

or 

```
> @USER And I said maybe etwas leiser singen, sonst ruf ich die Polizei
> @USER And I said vielleicht etwas leiser singen, sonst ruf ich die Polizei
```

Add the remove_chinese_space flag for Chinese--English text.

# Evaluate LLMs on the benchmark

You can evaluate the LLMs on the benchmark in the following style:

```
python acs/analysis/llms.py --langs de-en --incremental_lms meta-llama/Llama-3.1-8B --masked_lms FacebookAI/xlm-roberta-large
```

(any of the three arguments can be a space separated list)

Log-probabilities for each sentence in each minimal pair is saved by default in `data/results`. For the paired permutation-based significance tests, run them in the following style:

```
python acs/analysis/permutation_tests.py --langs de-en --lms bloom-560m xlm-roberta-large xlm-roberta-base
```

# Human judgments

All the collected human judgments are provided in `data/human_judgments`. `1` indicates that the participant selected the observed sentence, `0` indicates the participant selected the manipulated sentence. Run the agreement metrics with:

```
python acs/analysis/agreement.py
```

# Citation

The process is described in the following publication:

```
@inproceedings{sterner-2025-acs,
      author = {Igor Sterner and Simone Teufel},
      title = {Minimal Pair-Based Evaluation of Code-Switching},
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
}
```