# Emotion dataset for emotion classification tasks
import copy
import logging
import os
import pickle
from typing import Dict, Sequence
import io
import torch
from torch.utils.data import Dataset
import json
import hashlib

EMOTION_PROMPT_DICT = {
    "prompt_with_text": (
        "Below is a text that expresses certain emotions. "
        "Please classify the emotion expressed in this text.\n\n"
        "### Text:\n{text}\n\n### Emotion:"
    ),
    "prompt_simple": (
        "Text: {text}\nEmotion:"
    ),
}

# Emotion label mapping
EMOTION_LABELS = [" sadness", " joy", " love", " anger", " fear", " surprise"]
EMOTION_LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
EMOTION_ID_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_cache_path(data_path: str, tokenizer, prompt_type: str = None, split: str = None):
    """Generate cache file path based on data_path, tokenizer, and parameters."""
    # Create a hash of the tokenizer parameters and other relevant info
    tokenizer_info = {
        'vocab_size': tokenizer.vocab_size,
        'eod': tokenizer.eod,
        'name': getattr(tokenizer, 'name', 'unknown')
    }
    
    # Create hash from data_path and tokenizer info
    hash_input = f"{data_path}_{tokenizer_info}_{prompt_type}_{split}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Determine base directory and filename
    if os.path.isfile(data_path):
        base_dir = os.path.dirname(data_path)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
    else:
        # For HuggingFace datasets, use current directory
        base_dir = "."
        base_name = data_path.replace("/", "_")
    
    cache_filename = f"{base_name}_tokenized_{hash_value}.pkl"
    cache_path = os.path.join(base_dir, cache_filename)
    
    return cache_path

def load_cached_data(cache_path: str):
    """Load cached tokenized data."""
    # First check if cache file exists
    if not os.path.exists(cache_path):
        logging.warning(f"Cache file does not exist: {cache_path}")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        logging.warning(f"Loaded cached tokenized data from: {cache_path}")
        return cached_data
    except Exception as e:
        logging.warning(f"Failed to load cached data: {e}")
        return None

def save_cached_data(cache_path: str, data):
    """Save tokenized data to cache."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logging.warning(f"Saved tokenized data to cache: {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save cached data: {e}")

def load_data(data_path: str):
    """Load data from either HuggingFace dataset or JSON file."""
    # Check if data_path is a file path
    if os.path.isfile(data_path) or data_path.endswith('.json'):
        logging.warning(f"Loading data from JSON file: {data_path}")
        
        # Try to load as JSONL first (one JSON object per line)
        try:
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Invalid JSON on line {line_num}: {e}")
                            continue
            
            if data:
                logging.warning(f"Successfully loaded {len(data)} items from JSONL format")
                return data
            else:
                raise ValueError("No valid JSON objects found in file")
                
        except Exception as e:
            logging.warning(f"Failed to load as JSONL, trying as regular JSON: {e}")
            
            # Fallback to regular JSON format
            try:
                return jload(data_path)
            except Exception as e2:
                raise ValueError(f"Failed to load file as both JSONL and JSON: {e2}")
    else:
        # Assume it's a HuggingFace dataset name
        try:
            from datasets import load_dataset
            logging.warning(f"Loading data from HuggingFace dataset: {data_path}")
            dataset = load_dataset(data_path)
            
            # Handle different dataset splits
            if 'train' in dataset:
                data = dataset['train']
            elif 'validation' in dataset:
                data = dataset['validation']
            elif 'test' in dataset:
                data = dataset['test']
            else:
                # Use the first available split
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
            
            # Convert to list of dictionaries
            return [{"text": item["text"], "label": item["label"]} for item in data]
            
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{data_path}': {str(e)}")

class EmotionDataset(Dataset):
    """Dataset for emotion classification fine-tuning."""

    def __init__(self, data_path: str, tokenizer, prompt_type="prompt_with_text"):
        super(EmotionDataset, self).__init__()
        
        # Check for cached tokenized data
        cache_path = get_cache_path(data_path, tokenizer, prompt_type)
        cached_data = load_cached_data(cache_path)
        
        if cached_data is not None:
            self.text = cached_data["text"]
            self.input_ids_lens = cached_data["input_ids_lens"]
            self.label_ids = cached_data["label_ids"]
            self.label_text = cached_data["label_text"]
            logging.warning(f"Using cached tokenized data for {len(self.text)} samples")
        else:
            logging.warning("Loading emotion data...")
            list_data_dict = load_data(data_path)
            logging.warning("Formatting inputs...")
            
            prompt_template = EMOTION_PROMPT_DICT[prompt_type]
            sources = [
                prompt_template.format_map(example)
                for example in list_data_dict
            ]
            # Convert numeric label to emotion text
            targets = [f"{EMOTION_ID_TO_LABEL[example['label']]}" for example in list_data_dict]
            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)
            self.text = data_dict["text"]
            self.input_ids_lens = data_dict["input_ids_lens"]
            self.label_ids = data_dict["label_ids"]
            self.label_text = data_dict["label_text"]
            # Save to cache
            cache_data = {
                "text": self.text,
                "input_ids_lens": self.input_ids_lens,
                "label_ids": self.label_ids,
                "label_text": self.label_text
            }
            save_cached_data(cache_path, cache_data)
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(text=self.text[i], input_ids_lens=self.input_ids_lens[i], label_ids=self.label_ids[i], label_text=self.label_text[i])


def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = []
    for text in strings:
        # Use GPT2BPETokenizer's tokenize method
        token_ids = tokenizer.tokenize(text)
        # Convert to tensor and pad if needed
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        tokenized_list.append(token_ids)

    # Find max length for padding
    max_len = max(len(ids) for ids in tokenized_list)
    
    # Pad sequences
    padded_input_ids = []
    input_ids_lens = []
    for token_ids in tokenized_list:
        # Pad with eod token
        padding = torch.full((max_len - len(token_ids),), tokenizer.eod, dtype=torch.long)
        padded_ids = torch.cat([token_ids, padding])
        padded_input_ids.append(padded_ids)
        input_ids_lens.append(len(token_ids))
    return dict(
        text=padded_input_ids,
        input_ids_lens=input_ids_lens
    )
    
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, target_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, targets)]
    text = examples_tokenized["text"]
    label_ids = target_tokenized["text"]
    input_ids_lens = examples_tokenized["input_ids_lens"]
    return dict(text=text, input_ids_lens=input_ids_lens, label_ids=label_ids, label_text=targets)