import torch
from transformers import (
    BertModel, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PretrainedConfig,
    PreTrainedModel
)
from tqdm import tqdm
import os
from datasets import load_dataset,load_from_disk
import warnings
warnings.filterwarnings('ignore')

SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]']

class DualBertConfig(PretrainedConfig):
    model_type = 'dual_bert'
    def __init__(self, **kwargs):
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-base-chinese')
        super().__init__(**kwargs)

class DualBert(PreTrainedModel):
    config_class = DualBertConfig
    def __init__(self, config):
        super().__init__(config)
        self.bert1 = BertModel.from_pretrained(config.bert_model_name)
        self.bert2 = BertModel.from_pretrained(config.bert_model_name)

    def forward(self, input_text, candidate_text):
        batch_cls1 = self.bert1(**input_text).last_hidden_state[:, 0, :]
        batch_cls2 = self.bert2(**candidate_text).last_hidden_state[:, 0, :]
        similarity_scores = batch_cls1.mm(batch_cls2.T)
        return similarity_scores

def build_dataset(data_path, tokenizer):
    save_path = data_path.replace('data', 'processed_data').replace('.jsonl', '')
    if os.path.exists(save_path):
        return load_from_disk(save_path)
    
    dataset = load_dataset('json', data_files=data_path)['train']
    knowledge_base = load_dataset('csv', data_files='/home/luowenyang/wikiEL/data/wp_title_desc_wd0315.tsv', delimiter='\t', column_names=['qid', 'title', 'text'])['train']
    def find_match_index(qid, data_base):
        start = 0
        end = len(data_base)-1
        while start <= end:
            id = int(qid[1:])
            mid = start + (end-start)//2
            search_id = int(data_base[mid]['qid'][1:])
            
            if search_id == id:
                return mid
            elif search_id < id:
                start = mid + 1
            else:
                end = mid - 1
        return -1
    def preprocess(example):
        qid = example['gold_id']
        match_index = find_match_index(qid, knowledge_base)
        if match_index == -1:
            return {
                'input_text': '',
                'candidate_text': ''
            }
        try:
            context_left = example['text'][:example['start']]
            context_right = example['text'][example['end']:]
            input_text = tokenizer.cls_token + context_left + SPECIAL_TOKENS[0] + example['mention'] + SPECIAL_TOKENS[1] + context_right + tokenizer.sep_token
            candidate_text = tokenizer.cls_token + knowledge_base[match_index]['title'] + SPECIAL_TOKENS[2] + knowledge_base[match_index]['text'] + tokenizer.sep_token
            
            return {
                'input_text': input_text,
                'candidate_text': candidate_text
            }
        except:
            return {
                'input_text': '',
                'candidate_text': ''
            }
        
    def filter_empty(example):
        return len(example['input_text']) > 0
    dataset = dataset.map(preprocess, num_proc=24, remove_columns=['text', 'start', 'end', 'gold_id', 'mention','id'])
    dataset = dataset.filter(filter_empty, num_proc=24)
    if save_path:
        dataset.save_to_disk(save_path)
    return dataset

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_texts = [example['input_text'] for example in examples]
        candidate_texts = [example['candidate_text'] for example in examples]
        input_encodings = self.tokenizer(input_texts, return_tensors='pt', add_special_tokens=False, padding=True, max_length=128, truncation=True)
        candidate_encodings = self.tokenizer(candidate_texts, return_tensors='pt', add_special_tokens=False, padding=True, max_length=128, truncation=True)
        return input_encodings, candidate_encodings

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        input_encodings = inputs[0]
        candidate_encodings = inputs[1]
        similarity_scores = model(input_encodings, candidate_encodings)
        loss = torch.nn.functional.cross_entropy(similarity_scores, torch.arange(similarity_scores.shape[1]).to(similarity_scores.device))
        return loss

# Preprocess data
data_path = '/home/luowenyang/wikiEL/data/train-001.jsonl'

# Hyperparameters
batch_size = 96
lr = 5e-5
num_epochs = 1

# Initialize model and tokenizer
config = DualBertConfig(bert_model_name='bert-base-chinese')
model = DualBert(config)
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
model.bert1.resize_token_embeddings(len(tokenizer))
model.bert2.resize_token_embeddings(len(tokenizer))

# Prepare data loader
dataset = build_dataset(data_path, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    logging_dir="./logs",
    report_to="none",  # Don't report to any backend (like wandb or comet)
    save_strategy="epoch",
    save_steps=0.2,
)

# Training function
def training_function():
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained("./results")

# Start training
if __name__ == "__main__":
    training_function()
