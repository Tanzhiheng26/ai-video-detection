#%%
import pathlib
import os
import evaluate
import numpy as np
from utils import collate_fn 
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from dataloader import load_dataset
import json
#%%
with open("config.json") as f:
    config = json.load(f)

clip_duration = config["clip_duration"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
dataset_root_path = pathlib.Path(config["dataset_root_path"])
model_ckpt = config["base_model"]
dataloader_num_workers = config["dataloader_num_workers"]

#%%
label2id = {"real": 0, "fake": 1}
id2label = {0: "real", 1: "fake"}
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

#%%
train_dataset = load_dataset(
    data_path=os.path.join(dataset_root_path, "train"),
    model=model,
    image_processor=image_processor,
    clip_duration=clip_duration,
    training=True
)

val_dataset = load_dataset(
    data_path=os.path.join(dataset_root_path, "val"),
    model=model,
    image_processor=image_processor,
    clip_duration=clip_duration,
    training=False
)
#%%
model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned"

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="best",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    dataloader_num_workers=dataloader_num_workers
)
#%%
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
#%%
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
# %%
train_results = trainer.train()