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

def getResponse(chosen, rejected, pattern, training_mode):
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

    if training_mode == "sft":
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

    elif training_mode == "dpo":
        prompt = ""
        chosen = chosen.strip()
        rejected = rejected.strip()
        return prompt, chosen, rejected
    else:
        return None, None, None

def processFile(filepath, training_mode):
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
        prompt, chosen_text, rejected_text = getResponse(chosen, rejected, pattern, training_mode)

        if prompt == False: continue
        if prompt is not None:
            processed_data.append({
                'prompt': prompt,
                'chosen': chosen_text,
                'rejected': rejected_text
            })

    return pd.DataFrame(processed_data)

def createDatasets(folder, files, training_mode):
    """
    Create three separate DataFrames for train, validation, and test
    Returns: train_df, val_df, test_df
    """
    train_df = processFile(folder + files[0], training_mode)
    val_df = processFile(folder + files[1], training_mode)
    test_df = processFile(folder + files[2], training_mode)

    return train_df, val_df, test_df

def parse_conversation(text):
    """
    Parse a conversation string into message format
    """
    # Replace markers with delimiters
    text = text.replace("Human:", "|||HUMAN|||")
    text = text.replace("Assistant:", "|||ASSISTANT|||")
    
    parts = text.split("|||")
    
    messages = []
    current_role = None
    current_content = ""
    
    for part in parts:
        part = part.strip()
        
        if part == "HUMAN":
            # Save previous message if exists
            if current_role and current_content.strip():
                messages.append({
                    "role": current_role,
                    "content": current_content.strip()
                })
            current_role = "user"
            current_content = ""
        elif part == "ASSISTANT":
            # Save previous message if exists
            if current_role and current_content.strip():
                messages.append({
                    "role": current_role,
                    "content": current_content.strip()
                })
            current_role = "assistant"
            current_content = ""
        elif part:
            current_content += part
    
    # Don't forget the last message
    if current_role and current_content.strip():
        messages.append({
            "role": current_role,
            "content": current_content.strip()
        })
    
    return messages

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

class DPOPreprocessor:
    """
    Preprocessor for Direct Preference Optimization (DPO) data
    """
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True).copy()
    
    def build(self):
        """
        Build DPO dataset with chosen and rejected message pairs
        """
        examples = []
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            # Process chosen conversation
            chosen_messages = parse_conversation(chosen)
            
            # Process rejected conversation
            rejected_messages = parse_conversation(rejected)
            
            # Only add if both conversations parsed successfully
            if chosen_messages and rejected_messages:
                examples.append({
                    "chosen": chosen_messages,
                    "rejected": rejected_messages
                })
        
        return examples

def get_sft_datasets():
    """
    Get the SFT datasets
    """
    train_df, val_df, test_df = createDatasets(FOLDER, [TRAIN_FILE, VAL_FILE, TEST_FILE], "sft")
    train_dataset = SFTPreprocessor(train_df).build()
    val_dataset = SFTPreprocessor(val_df).build()
    test_dataset = SFTPreprocessor(test_df).build()

    # convert to HuggingFace Dataset objects
    train_dataset = HFDataset.from_list(train_dataset)
    val_dataset = HFDataset.from_list(val_dataset)
    test_dataset = HFDataset.from_list(test_dataset)
    return train_dataset, val_dataset, test_dataset

def get_dpo_datasets():
    """
    Get the DPO datasets
    """
    train_df, val_df, test_df = createDatasets(FOLDER, [TRAIN_FILE, VAL_FILE, TEST_FILE], "dpo") # pandas dataframe

    train_dataset = DPOPreprocessor(train_df).build() # to list
    val_dataset = DPOPreprocessor(val_df).build()
    test_dataset = DPOPreprocessor(test_df).build()

    # convert to HuggingFace Dataset objects
    train_dataset = HFDataset.from_list(train_dataset) # to huggingface dataset
    val_dataset = HFDataset.from_list(val_dataset)
    test_dataset = HFDataset.from_list(test_dataset)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_dpo_datasets()



