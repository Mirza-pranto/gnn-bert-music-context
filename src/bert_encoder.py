import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_MultiLabelClassifier(nn.Module):
    def __init__(self, num_tags, model_name='bert-base-uncased', freeze_bert=False):
        super(BERT_MultiLabelClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"BERT initialized with {trainable_params:,} trainable parameters (total: {total_params:,})")
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        # Extract the outputs from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        
        cls_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
   
        logits = self.classifier(cls_output)
        
        if return_embeddings:
            return logits, cls_output, last_hidden_state
        
        return logits

def get_tokenizer(model_name='bert-base-uncased'):
    return BertTokenizer.from_pretrained(model_name)
