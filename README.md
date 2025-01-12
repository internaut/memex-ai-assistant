# AI assistant for encrypted *memex* notes

January 2025

Author: Markus Konrad <post@mkonrad.net>

## Description

This repository contains a Python script for creating and chatting with an AI assistant about your personal, encrypted [memex](https://github.com/internaut/memex) notes.

It uses a Retrieval-augmented generation (RAG) approach, where at first all relevant notes for a specific query are retrieved using a [CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model. These notes provide the context for an LLM to answer user queries. This project uses [llama.cpp bindings for Python](https://llama-cpp-python.readthedocs.io/) to load a Llama-based LLM. The default LLM is the 8 bit quantized [Llama-3.2-1B-Instruct-Q8 model](https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF) model. 

## Usage

    usage: ask_memex.py [-h] [--key KEY] [--keypw KEYPW] [--memexdb MEMEXDB] [--num-context-docs NUM_CONTEXT_DOCS] [--num-context-tokens NUM_CONTEXT_TOKENS] [--num-workers NUM_WORKERS] [--sample SAMPLE] [--encoding ENCODING] [--verbose] [--use-cache] [query]
    
    positional arguments:
      query                 initial query; leave empty to be prompted
    
    options:
      -h, --help            show this help message and exit
      --key KEY             path to private GPG key file in ASC format
      --keypw KEYPW         private GPG key password; leave empty to be prompted when key is locked
      --memexdb MEMEXDB     path to memex files
      --num-context-docs NUM_CONTEXT_DOCS
                            number of memex notes that make up the RAG context
      --num-context-tokens NUM_CONTEXT_TOKENS
                            maximum number of input tokens
      --num-workers NUM_WORKERS
                            number of worker processes used for parallel processing
      --sample SAMPLE       draw only a sample of the memex notes by specifying the sample size
      --encoding ENCODING   memex notes encoding
      --verbose             turn on verbose output
      --use-cache           cache decrypted notes in temp. directory; use with care!
