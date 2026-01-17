import torch
from torch.utils.data import Dataset, DataLoader
import trl
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset

"""
PARSING
"""
import re

FOLDER = "data/"
TRAIN_FILE = "hh_rlhf_train_10k.csv"
VAL_FILE = "hh_rlhf_val_3k.csv"
TEST_FILE = "hh_rlhf_test_1k.csv"

def getResponse(chosen, rejected, pattern):
    """
    Given a list of Human and Assistant conversation we want to parse out the prompt,
    chosen, and rejected
    """
    # match all Human/Assistant pairs
    chosen_matches = re.findall(pattern, chosen, re.DOTALL)
    rejected_matches = re.findall(pattern, rejected, re.DOTALL)

    # to prevent index out of bounds :(
    word = "Assistant"
    chosen_count = len(re.findall(rf"\b{word}\b", chosen))
    rejected_count = len(re.findall(rf"\b{word}\b", rejected))

    if chosen_count != rejected_count: return False, False, False
    if not chosen_matches:
        return None, None, None

    # build conversation until last Assistant response
    prompt_parts = []
    last_assistant_idx = -1

    # find the last Assistant response
    for i, (role, text, _) in enumerate(chosen_matches):
        if role == "Assistant":
            last_assistant_idx = i

    # cannot find 'Assistant'
    if last_assistant_idx == -1:
        return None, None, None

    # everything before last Assistant is the prompt
    for i in range(last_assistant_idx):
        role, text, _ = chosen_matches[i]
        prompt_parts.append(f"{role}: {text.strip()}")

    prompt = "\n\n".join(prompt_parts)

    # last Assistant response is the chosen/rejected (depends on your data structure)
    _, chosen_response, _ = chosen_matches[last_assistant_idx]
    _, rejected_response, _ = rejected_matches[last_assistant_idx]
    chosen, rejected = chosen_response.strip(), rejected_response.strip()

    return prompt, chosen, rejected

def processFile(filepath):
    """
    Process a single CSV file and return a DataFrame
    """
    # read the csv
    df = pd.read_csv(filepath)

    # match 'Human:' or 'Assistant:' followed by any text until the next role or end of string
    pattern = r"(Human|Assistant):\s*(.*?)(?=(Human|Assistant):|$)"

    processed_data = []

    for idx, row in df.iterrows():
        chosen, rejected = row["chosen"], row["rejected"]
        prompt, chosen_text, rejected_text = getResponse(chosen, rejected, pattern)

        if prompt == False: continue
        if prompt is not None:
            processed_data.append({
                'prompt': prompt,
                'chosen': chosen_text,
                'rejected': rejected_text
            })

    return pd.DataFrame(processed_data)

def createDatasets(folder, files):
    """
    Create three separate DataFrames for train, validation, and test
    Returns: train_df, val_df, test_df
    """
    train_df = processFile(folder + files[0])
    val_df = processFile(folder + files[1])
    test_df = processFile(folder + files[2])

    return train_df, val_df, test_df

"""
Creating the Dataset class
"""
class SFTPreprocessor:
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True).copy()

    def build(self):
        examples = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            full_text = str(row["prompt"]) + "\n\nAssistant: " + str(row["chosen"])
            
            full_text = full_text.replace("Human:", "|||HUMAN|||")
            full_text = full_text.replace("Assistant:", "|||ASSISTANT|||")
            parts = full_text.split("|||")
            
            messages = []
            current_role = None
            current_content = ""
            
            for part in parts:
                part = part.strip()
                if part == "HUMAN":
                    if current_role and current_content.strip():
                        messages.append(
                            {"role": current_role, "content": current_content.strip()}
                        )
                    current_role = "user"
                    current_content = ""
                elif part == "ASSISTANT":
                    # Save previous message if exists
                    if current_role and current_content.strip():
                        messages.append(
                            {"role": current_role, "content": current_content.strip()}
                        )
                    current_role = "assistant"
                    current_content = ""
                elif part:
                    current_content += part
            
            if current_role and current_content.strip():
                messages.append(
                    {"role": current_role, "content": current_content.strip()}
                )
            
            if messages:
                examples.append({"messages": messages})
        
        return examples

def get_sft_datasets():
    """
    Get the SFT datasets
    """
    train_df, val_df, test_df = createDatasets(FOLDER, [TRAIN_FILE, VAL_FILE, TEST_FILE])
    train_dataset = SFTPreprocessor(train_df).build()
    val_dataset = SFTPreprocessor(val_df).build()
    test_dataset = SFTPreprocessor(test_df).build()

    # convert to HuggingFace Dataset objects
    train_dataset = HFDataset.from_list(train_dataset)
    val_dataset = HFDataset.from_list(val_dataset)
    test_dataset = HFDataset.from_list(test_dataset)
    return train_dataset, val_dataset, test_dataset












class DPODataset(Dataset):
    """DPO dataset without prompts — only chosen and rejected completions."""

    def __init__(self, dataframe, tokenizer, max_length):
        """
        Args:
            dataframe: pandas DataFrame with 'prompt', 'chosen', 'rejected' columns
            tokenizer: HuggingFace tokenizer
            max_length: maximum sequence length for tokenization
        """
        self.data = dataframe.reset_index(drop=True)  # use the dataframe directly
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str):
        """Tokenize a single text string."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset."""
        row = self.data.iloc[idx]

        # get individual components
        prompt = str(row['prompt'])
        chosen = str(row['chosen'])
        rejected = str(row['rejected'])

        # tokenize each separately
        prompt_tokens = self._tokenize(prompt)
        chosen_tokens = self._tokenize(chosen)
        rejected_tokens = self._tokenize(rejected)

        return {
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0),
        }

