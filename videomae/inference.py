# %%
import pathlib
import os
import numpy as np
from utils import collate_fn 
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from dataloader import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# %%
dataset_root_path = pathlib.Path("../data")
label2id = {"real": 0, "fake": 1}
id2label = {0: "real", 1: "fake"}
model_ckpt = "checkpoints/checkpoint-1000"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
clip_duration = 5

# %%
test_dataset = load_dataset(
    data_path=os.path.join(dataset_root_path, "test"),
    model=model,
    image_processor=image_processor,
    clip_duration=clip_duration,
    training=False
)

# %%
args = TrainingArguments(
    remove_unused_columns=False,
)

trainer = Trainer(
    model,
    args,
    processing_class=image_processor,
    data_collator=collate_fn,
)

# %%
test_pred = trainer.predict(test_dataset)

# %%
y_pred = np.argmax(test_pred.predictions, axis=1)
y_true = test_pred.label_ids
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

# %%
acc = accuracy_score(y_true, y_pred)
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Accuracy:", acc)
print("Precision:", p)
print("Recall", r)
print("F1 score:", f1)
