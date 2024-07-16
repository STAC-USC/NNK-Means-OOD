from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset, DatasetDict, ClassLabel, load_dataset, concatenate_datasets, Value
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def tokenize(batch, tokenizer):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True)

def load_clinic150(file_path):
    '''
    Load CLINC150 dataset from a local JSON file.

    Original split:
    - 'train': training set for is
    - 'oos_train': training set for oos
    - 'val': validation set for is
    - 'oos_val': validation set for oos
    - 'test': test set for is
    - 'oos_test': test set for oos

    Our split:
    - Train: original 'train' set
    - Val: original 'val' set + 'oos_val' set
    - Test: original 'test' set + 'oos_test' set

    Return:
    - processed_dataset: DatasetDict
    - known_labels: list of known labels
    '''
    print("\nLoading CLINC150 dataset...")

    processed_dataset = DatasetDict()

    # Load the dataset from the JSON file
    with open(file_path, 'r') as f:
        raw_dataset_clinc150 = json.load(f)

    # Prepare the train dataset
    train_data = [item for item in raw_dataset_clinc150['train']]
    train_sentences = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]

    # Prepare the val dataset
    val_data = raw_dataset_clinc150['oos_val'] + raw_dataset_clinc150['val']
    val_sentences = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]

    # Prepare the test dataset
    test_data = raw_dataset_clinc150['oos_test'] + raw_dataset_clinc150['test']
    test_sentences = [item[0] for item in test_data]
    test_labels = [item[1] for item in test_data]

    # Create a mapping from label strings to integers
    all_labels = list(set(train_labels + val_labels + test_labels))
    label2id = {label: idx for idx, label in enumerate(all_labels)}

    # Convert labels to integers
    train_labels = [label2id[label] for label in train_labels]
    val_labels = [label2id[label] for label in val_labels]
    test_labels = [label2id[label] for label in test_labels]

    # Convert to Dataset
    train_dataset = Dataset.from_dict({
        'sentence': train_sentences,
        'label': train_labels
    })
    val_dataset = Dataset.from_dict({
        'sentence': val_sentences,
        'label': val_labels
    })
    test_dataset = Dataset.from_dict({
        'sentence': test_sentences,
        'label': test_labels
    })
    processed_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    # Process train dataset to only keep known labels
    known_labels = list(set(train_labels))

    # Filter the training dataset to only include samples with known labels
    processed_dataset['train'] = processed_dataset['train'].filter(lambda x: x['label'] in known_labels)

    print(f"Num. of known labels: {len(known_labels)}")
    print(f"Original known labels: {known_labels}")
    print(f"Train Labels: {list(set(train_labels))}")
    print(f"Val Labels: {list(set(val_labels))}")
    print("CLINC150 dataset loaded.\n")

    return processed_dataset, known_labels

def load_data_banking(known_ratio=0.25):
    '''
    Load Banking77 dataset from HuggingFace datasets.

    Original split:
    - 'train': 10003
    - 'test': 3080

    Our split:
    - Train & Val: val is splited from 'train' by 0.1
    - Test: original 'test' split

    Return:
    - processed_dataset: DatasetDict
    - known_labels: list of known labels
    '''
    print("\nLoading Banking77 dataset...")

    processed_dataset = DatasetDict()

    raw_dataset_banking = load_dataset('PolyAI/banking77')
    
    # Prepare the dataset
    extracted_dataset = []
    for i in range(len(raw_dataset_banking['train'])):
        sentence = raw_dataset_banking['train'][i]['text']  # str
        label = raw_dataset_banking['train'][i]['label']  # int
        extracted_dataset.append({'sentence': sentence, 'label': label})

    # Convert to Dataset
    train_dataset = Dataset.from_dict({
        'sentence': [x['sentence'] for x in extracted_dataset],
        'label': [x['label'] for x in extracted_dataset]
    })
    
    # Convert the labels column to ClassLabel type
    class_labels = ClassLabel(names=raw_dataset_banking['train'].features['label'].names)
    train_dataset = train_dataset.cast_column('label', class_labels)

    # Split into train and val
    train_val_split = train_dataset.train_test_split(test_size=0.1, stratify_by_column='label')

    # Use the original test set
    test_dataset = raw_dataset_banking['test']
    test_dataset = Dataset.from_dict({
        'sentence': [x['text'] for x in test_dataset],
        'label': [x['label'] for x in test_dataset]
    })
    test_dataset = test_dataset.cast_column('label', class_labels)

    processed_dataset = DatasetDict({
        'train': train_val_split['train'],
        'val': train_val_split['test'],
        'test': test_dataset
    })

    # Process train dataset to only keep known labels
    all_labels = list(set(processed_dataset['train']['label']))
    known_labels = np.random.choice(all_labels, int(known_ratio * len(all_labels)), replace=False)

    # Filter the training dataset to only include samples with known labels
    processed_dataset['train'] = processed_dataset['train'].filter(lambda x: x['label'] in known_labels)


    print(f"Num. of known labels: {len(known_labels)}")
    print(f"Original known labels: {known_labels}")
    print("Banking77 dataset loaded.\n")

    return processed_dataset, known_labels

def load_data_20ng(known_ratio=0.25):
    '''
    Load 20 Newsgroups dataset from sklearn.datasets.

    Original split:
    - 'train': 11314
    - 'test': 7532
    - 'all': 18846

    Our split: we use 'all' and then do 80/10/10 split.

    Return:
    - processed_dataset: DatasetDict
    - known_labels: list of known labels
    '''
    print("\nLoading 20 Newsgroups dataset...")

    processed_dataset = DatasetDict()

    raw_dataset_20ng = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    len(raw_dataset_20ng.data)

    # Prepare the dataset
    extracted_dataset = []
    for i in range(len(raw_dataset_20ng.data)):
        sentence = raw_dataset_20ng.data[i]  # str
        label = raw_dataset_20ng.target[i]  # numpy.int64
        label_text = raw_dataset_20ng.target_names[label]  # str
        extracted_dataset.append({'sentence': sentence, 'label': label, 'label_text': label_text})

    # Convert to Dataset
    full_dataset = Dataset.from_dict({
        'sentence': [x['sentence'] for x in extracted_dataset],
        'label': [x['label'] for x in extracted_dataset],
        'label_text': [x['label_text'] for x in extracted_dataset]
    })
    
    # Convert the labels column to ClassLabel type
    class_labels = ClassLabel(names=raw_dataset_20ng.target_names)
    full_dataset = full_dataset.cast_column('label', class_labels)

    # Split into train, val, test
    train_val_test_split = full_dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, stratify_by_column='label')

    processed_dataset = DatasetDict({
        'train': train_val_test_split['train'],
        'val': val_test_split['train'],
        'test': val_test_split['test']
    })

    # Process train dataset to only keep known labels
    all_labels = list(set(processed_dataset['train']['label']))
    known_labels = np.random.choice(all_labels, int(known_ratio * len(all_labels)), replace=False)

    # Filter the training dataset to only include samples with known labels
    processed_dataset['train'] = processed_dataset['train'].filter(lambda x: x['label'] in known_labels)

    print(f"Num. of known labels: {len(known_labels)}")
    print(f"Original known labels: {known_labels}")
    print("20 Newsgroups dataset loaded.\n")

    return processed_dataset, known_labels

def preprocess_data(processed_dataset, known_labels, tokenizer):
    '''
    Tokenizes and formats the dataset for training
    Args:
        - processed_dataset: DatasetDict containing train, val, and test splits of data. Each split contains:
            - sentence: list of sentences
            - label: list of labels
            - label_text: list of label names
        - known_labels: list of labels chosen to be known / in-distribution
    Returns:
    Dataloaders for train, val, and test data with tokenized and formatted data
        - train_dataloader: DataLoader for training data
        - val_dataloader: DataLoader for validation data
        - test_dataloader: DataLoader for test data
    '''
    train_dataset = processed_dataset['train']
    val_dataset = processed_dataset['val']
    test_dataset = processed_dataset['test']

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=128, fn_kwargs={'tokenizer': tokenizer})
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=128, fn_kwargs={'tokenizer': tokenizer})
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=128, fn_kwargs={'tokenizer': tokenizer})

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'], output_all_columns=True)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def get_sent_embedding(model, batch, device):
    output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
    last_hidden_states = output.last_hidden_state
    pooled_output = torch.mean(last_hidden_states, dim=1)
    return pooled_output 

def extract_embeddings(train_dataloader, test_dataloader, device, model, known_labels):
    '''
    Extract embeddings for train and test data
    Args:
        - train_dataloader: DataLoader for training data
        - test_dataloader: DataLoader for test data
        - device: device to run on
        - model: Embedding model (e.g. BERT)
    Returns:
        - bank: embeddings for train data
        - label_bank: labels for train data
        - test_bank: embeddings for test data
        - ood_labels: labels for test data (1 if known, 0 if unknown)
    '''
    model = model.to(device)
    bank = None
    label_bank = None

    for batch in tqdm(train_dataloader):
        pooled = get_sent_embedding(model, batch, device)
        if bank is None:
            bank = pooled.clone().detach()
            label_bank = batch['label']
        else:
            bank = torch.cat((bank, pooled.clone().detach()), dim=0)
            label_bank = torch.cat((label_bank, batch['label']), dim=0)

    test_bank = None
    ood_labels = None
    test_labels = None

    for batch in tqdm(test_dataloader):
        pooled = get_sent_embedding(model, batch, device)
        if test_bank is None:
            test_bank = pooled.clone().detach()
            # set ood_labels to 1 if the label is in the known labels and 0 otherwise
            ood_labels = torch.zeros(pooled.size(0))
            test_labels = torch.zeros(pooled.size(0))
            for i, label in enumerate(batch['label']):
                if label.item() in known_labels:
                    ood_labels[i] = 1
                test_labels[i] = label.item()
                    
        else:
            test_bank = torch.cat((test_bank, pooled.clone().detach()), dim=0)
            temp_ood_labels = torch.zeros(pooled.size(0))
            temp_test_labels = torch.zeros(pooled.size(0))
            for i, label in enumerate(batch['label']):
                if label.item() in known_labels:
                    temp_ood_labels[i] = 1
                temp_test_labels[i] = label.item()
            ood_labels = torch.cat((ood_labels, temp_ood_labels), dim=0)
            test_labels = torch.cat((test_labels, temp_test_labels), dim=0)
    return bank, label_bank, test_bank, ood_labels, test_labels