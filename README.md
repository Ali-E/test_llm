diff --git a/README.md b/README.md
index 53bfbb4e214b6ab5d30d189a18283172c56b59ed..51fda4e1c1f88a5fa96e41d2f6f1de18e33db27b 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,19 @@
-loading a pretrained model like gpt2 from transformers library. then finetune it on harry potter books from the MUSE paper to create a reinforced model and save the new model.
+This project fine-tunes the GPT-2 language model on a corpus of Harry Potter text (as used in the MUSE paper).
+
+## Setup
+Install the required packages (PyTorch, transformers and datasets). If you do not already have them, run:
+
+```bash
+pip install torch transformers datasets
+```
+
+A small sample of text is provided in `data/harry_potter_sample.txt` for demonstration purposes. Replace this with the full dataset if available.
+
+## Training
+Run the training script:
+
+```bash
+python train.py
+```
+
+The script loads `gpt2`, fine-tunes it on the text data and saves the resulting model to the `reinforced_model` directory.

