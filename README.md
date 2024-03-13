## Parallel Multilingual Pre-training for Multimodal Representations
Multi-modal representations are useful in many downstream tasks such as image-grounded caption generation and text-grounded image generation. The majority of the datasets available for training these representations are in English or are heavily skewed to certain languages. We extend the CLIP technique from Radford et al. (2021) by incorporating artificial parallel data from three additional diverse languages and find that this not only improves the multilingual performance of the CLIP model, but also improves its performance in English.

### File Descriptions

#### translate_data.py
This takes a dictionary of captions and translates them into more languages (to create n-way parallel data)

#### finetune_clip.py
This trains the text and image encoders to merge their distributions.

#### generate_vectors.py
This generates vectors from your data that you can use for downstream tasks or evaluation.
