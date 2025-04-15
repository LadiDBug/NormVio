# Introduction:
This fork introduces several modifications to the original training script to simplify the setup and specifically support the +Community variant proposed in the original paper (Section 3.1).

In this variant, the model receives both the community information and the final comment as input. To incorporate community-specific signals, the subreddit name is prepended to the comment (e.g., "r/AskReddit ask anything!").

# Modifications:
1. Removed Rule-Based Supervision
    The `--path_rule argument` is now optional and no longer required for training. All related logic (e.g., df_rules) has been commented out, and the rule data is replaced with `None` during preprocessing.

2. Category Mapping via JSON
    A new `mappings/` directory has been added, containing the file `cat10_to_idx.json`, which maps category names to class indices.
    Structure of cat10_to_idx.json:
    {
    "off-topic": 1,
    "spam": 2,
    ...
    }

3. GPU Usage and Validagtion
    All model components (encoder, context encoder, classifier) are explicitly moved to the correct device using `.to(device)`. Additionally, runtime checks confirm GPU availability and print the name of the GPU being used. 

# How to Run:
Here is an example command to train the model `+community`:
```
python ./src/train_derailment_model_binary.py \
  --path_data [PATH_TO_DATASET] \
  --path_save [OUTPUT_DIR] \
  --append_subreddit `subreddit` \
  --batch_size [YOUR_BATCH_SIZE] \
  --valid_batch_size [YOUR_VALID_BATCH_SIZE] \
``` 
You can adjust other parameters (e.g number of epoch, early stopping) as needed.

# Training Duration:
Thanks to early stopping (default patience: 5), each model typically completes training in approximately 2 hours, even though the configuration allows for a maximum runtime of around 30 hours.
Training across all categories usually takes 20–30 hours in total.

