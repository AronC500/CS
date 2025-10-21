import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, TrainingArguments, TrainerCallback
from transformers import DataCollatorForLanguageModeling

from torch.utils.data import random_split, DataLoader, Dataset
import os 

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


 
dataset = torch.load('raworg2_retrained_bert_dataset.pt')
input_ids, attention_masks = dataset.tensors

 
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)

# Assuming `input_ids` and `attention_masks` are already loaded and on the correct device
encodings = {'input_ids': input_ids, 'attention_mask': attention_masks}
full_dataset = CustomDataset(encodings)
 
subset_size = int(0.01 * len(full_dataset))  #  
indices = torch.randperm(len(full_dataset))[:subset_size].tolist()

 
train_size = int(0.9 * len(indices))  #  
val_size = len(indices) - train_size

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)



 
config = BertConfig(
    vocab_size=30522,  #  
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=3072,
    hidden_act='gelu',
    max_position_embeddings=512
)

 
model = BertForMaskedLM(config).to(device)



class CustomTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch}: Training Loss: {state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else 'N/A'}, " \
              f"Validation Loss: {state.log_history[-1]['eval_loss'] if 'eval_loss' in state.log_history[-1] else 'N/A'}, " \
              f"Validation MLM Accuracy: {state.log_history[-1]['eval_accuracy'] if 'eval_accuracy' in state.log_history[-1] else 'N/A'}")
        
        model.save_pretrained(f'./bert_pretrained_512-88-pretrain/bert_pretrained_epoch_{int(state.epoch)}')



training_args = TrainingArguments(
    output_dir='./bert_pretrained_512-88',
    num_train_epochs=50,
    per_device_train_batch_size=16,
    evaluation_strategy='epoch',   
    save_strategy='epoch',   
    save_total_limit=50,  
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,   
    metric_for_best_model='eval_loss'   
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[CustomTrainerCallback()]
)

trainer.train()
  
model.save_pretrained('./final_bert_model')


