This project fine-tunes the GPT-2 language model on a corpus of Harry Potter text (as used in the MUSE paper).

## Setup
Install the required packages (PyTorch, transformers and datasets). If you do not already have them, run:

```bash
pip install torch transformers datasets
```

A small sample of text is provided in `data/harry_potter_sample.txt` for demonstration purposes. Replace this with the full dataset if available.

## Training
Run the training script:

```bash
python train.py
```

The script loads `gpt2`, fine-tunes it on the text data and saves the resulting model to the `reinforced_model` directory.
