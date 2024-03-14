import pandas as pd
import re
import random
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate

random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

df = pd.read_csv("data.tsv", sep='\t', header=0)

df = df.drop_duplicates(subset=df.columns[1])


sentence = "Oh, there's a butterfly!"

def replace_words(sentence, replacements):
    pattern = r'\b(' + '|'.join(re.escape(word) for word in replacements.keys()) + r')\b'
    def repl(match):
        return random.choice(replacements[match.group(0)])
    return re.sub(pattern, repl, sentence)

replacements = {"there": ["they're", "their"], "their": ["they're", "there"], "they're": ["there", "their"],
                "There": ["They're", "Their"], "Their": ["They're", "There"], "They're": ["There", "Their"]}

new_sentence = replace_words(sentence, replacements)

pattern = r'\b(' + '|'.join(re.escape(word) for word in replacements.keys()) + r')\b'

print(new_sentence)

df_filtered = df[df.columns[1]]
df_filtered = df_filtered[df_filtered.apply(lambda x: len(re.findall(pattern, x)) > 0)]
df_filtered = pd.DataFrame(df_filtered)
df_filtered = df_filtered.rename(columns={df_filtered.columns[0]: 'correct'})
df_filtered['incorrect'] = df_filtered['correct'].apply(lambda x: replace_words(x, replacements))
df_filtered.columns
df = df_filtered


model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

train_df, test_df = train_test_split(df, test_size=0.10, shuffle=True)


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

from torch.utils.data import Dataset, DataLoader

class GrammarDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.max_len = 64

    def __len__(self):
        return len(self.dataset)


    def tokenize_data(self, example):
        input_, target_ = example['incorrect'], example['correct']

        # tokenize inputs
        tokenized_inputs = tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
                                            max_length=self.max_len,
                                            return_attention_mask=True,
                                     truncation=True)

        tokenized_targets = tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
                                            max_length=self.max_len,
                                            return_attention_mask=True,
                                      truncation=True)

        inputs={"input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask'],
            "labels": tokenized_targets['input_ids']
        }

        return inputs


    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        return inputs


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')

# defining training related arguments
batch_size = 8
args = Seq2SeqTrainingArguments(output_dir="./output_dir/",
                        evaluation_strategy="steps",
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=1e-5,
                        num_train_epochs=3,
                        predict_with_generate=True,
                        fp16 = True,
                        gradient_accumulation_steps = 1,
                        )


bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    score = {'bleu': result['bleu']}
    return score

trainer = Seq2SeqTrainer(model=model,
                args=args,
                train_dataset= GrammarDataset(train_dataset, tokenizer),
                eval_dataset=GrammarDataset(test_dataset, tokenizer),
                tokenizer=tokenizer,
                data_collator=data_collator,
                )#compute_metrics=compute_metrics)

trainer.train()

model.eval()


def evaluate_single(input_text):
    batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=64, return_tensors="pt").to(device)
    translated = model.generate(**batch,max_length=64,num_beams=4,max_new_tokens=64, num_return_sequences=1, temperature=0.1)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def evaluate_all(input_texts, batch_size):
    # Batch tokenize input texts
    #input_texts = ["Correction: " + x for x in input_texts]
    outputs = []
    n_batches = len(input_texts)//batch_size
    #print(len(input_texts), n_batches)    
    for i in range(n_batches):
        batch = tokenizer(input_texts[i*batch_size:(i+1)*batch_size], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(device)
        translated = model.generate(**batch, max_length=64, num_beams=4, max_new_tokens=64, num_return_sequences=1, temperature=0.1)
        # Decode generated sequences
        tgt_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        #print(input_texts[i*batch_size:(i+1)*batch_size])
        #print(tgt_texts)
        #print(len(tgt_texts), len(input_texts[i*batch_size:(i+1)*batch_size]))
        for j in range(batch_size):
            outputs.append(tgt_texts[j])    
    
    if len(input_texts)%batch_size > 0:
        batch = tokenizer(input_texts[n_batches*batch_size:], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(device)
        translated = model.generate(**batch, max_length=64, num_beams=4, max_new_tokens=64, num_return_sequences=1, temperature=0.1)
        # Decode generated sequences
        tgt_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for j in range(len(tgt_texts)):
            outputs.append(tgt_texts[j])
    return outputs        

inputs = list(test_df['incorrect'])
outputs = evaluate_all(inputs, 64)
correct_outputs = list(test_df['correct'])
#print(outputs)
#print(len(outputs))

def accuracy(s1, s2):
    count = 0
    for i in range(len(s1)):
        if s1[i].strip() == s2[i].strip():
            count += 1
    return count/len(s1)        

#for i in range(100):
#    print("Input:{}   Output:{}".format(inputs[i], outputs[i]))

score = bleu.compute(predictions=outputs, references=correct_outputs)
print(score)
print(accuracy(outputs, correct_outputs))

