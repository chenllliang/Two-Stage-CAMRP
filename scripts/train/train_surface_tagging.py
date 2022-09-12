import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer
from datasets import Dataset,DatasetDict,load_metric
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")


def transform_files_to_split_dataset(token_path,tag_path):
    tokens = open(token_path,"r").readlines()
    tags = open(tag_path,"r").readlines()
    result = {"concept_tags":[],"tokens":[]}

    for tok,tag in zip(tokens,tags):
        cur_tok = tok.strip().split()
        cur_tag = [int(i) for i in tag.strip().split()]

        if len(cur_tok)!=len(cur_tag) and len(cur_tok) - len(cur_tag)==-1:
            cur_tag=cur_tag[:len(cur_tok)]
        
        if len(cur_tok)!=len(cur_tag):
            print(cur_tag,cur_tok)

        result["concept_tags"].append(cur_tag)
        result["tokens"].append(cur_tok)
        
    return Dataset.from_dict(result)

def load_inference_dataset(token_path):
    tokens = open(token_path,"r").readlines()
    result = {"concept_tags":[],"tokens":[]}
    for tok in tokens:
        cur_tok = tok.strip().split()
        cur_tag = [0]*len(cur_tok)


        result["tokens"].append(cur_tok)
        result["concept_tags"].append(cur_tag)
    return Dataset.from_dict(result)


def build_tagging_dataset_for_camr(train,dev):
    train=transform_files_to_split_dataset(train+".sent",train+".tag")
    dev=transform_files_to_split_dataset(dev+".sent",dev+".tag")
    return DatasetDict({"train":train,"dev":dev})


def build_extra_nodes_tagging_dataset_for_camr(train,dev):
    train=transform_files_to_split_dataset(train+".sent",train+".extra_nodes.tag")
    dev=transform_files_to_split_dataset(dev+".sent",dev+".extra_nodes.tag")
    return DatasetDict({"train":train,"dev":dev})

def build_concept_normalization_tagging_dataset_for_camr(train,dev):
    train=transform_files_to_split_dataset(train+".sent",train+".p_transform_tag")
    dev=transform_files_to_split_dataset(dev+".sent",dev+".p_transform_tag")
    return DatasetDict({"train":train,"dev":dev})

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"concept_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



camr_tag_dataset = build_tagging_dataset_for_camr("../../preprocessed/surface_tagging/train","../../preprocessed/surface_tagging/dev")

tokenized_dataset = camr_tag_dataset.map(tokenize_and_align_labels, batched=True)


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=8)
#model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=185)
metric = load_metric("seqeval")

# label_list=["O","B-NODE","B-Continue-Node","B-UnContinue-Node"]
# label_dict={"O":"0","B-NODE":"1","B-Continue-Node":"2","B-UnContinue-Node":"3"}

#6 classes 
# label_list=["O","B-NODE","I-NODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]
# label_dict={j:str(i) for i,j in enumerate(label_list)}

#7 classes 
#label_list=["O","B-EASY_NODE","B-CNODE","I-CNODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT"]

#8 classes 
label_list=["O","B-EASY_NODE","B-CNODE","I-CNODE","B-UnContinue-Node","I-UnContinue-Node","B-SPLIT","B-VIRTUAL"]


#185 classes
# tag_path="/home/cl/ATP/amrlib/amrlib/models/Unified_Parsing/CAMR/Tagging/datasets/v2_6classes/train.tag.extra_nodes_dict"
# tags_index={}
# with open(tag_path,"r") as f:
#     lines = f.readlines()
#     for i,j in enumerate(lines):
#         tags_index[j.strip()]=i+1

# label_list=["O"]+["B-"+i for i in tags_index.keys()]


#2 classes
#label_list=["O","B-Transform"]

label_dict={j:str(i) for i,j in enumerate(label_list)}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


OUT_DIR="../../models/surface_tagging"

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    learning_rate=2e-5,
    warmup_ratio=0.01,
    lr_scheduler_type="linear",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    num_train_epochs=50,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

predictions, labels, _ = trainer.predict(tokenized_dataset["dev"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

with open(OUT_DIR+"/best_output.txt","w") as f:
    for i in true_predictions:
        f.write(" ".join([label_dict[j] for j in i])+"\n")

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
