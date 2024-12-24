---
title: "Exploring Knowledge Distillation in Large Language Models"
date: 2023-11-15
categories: [Distillation]
tags: [Distillation, Bert, Huggingface, Transformers]
---




AI companies continue to scale language models, yet deploying these large models remains challenging. Knowledge distillation is one way to have models efficient in production.

**Knowledge distillation** transfers knowledge from a large "teacher" model to a smaller "student" model. This method reduces computational demands and enables practical applications, especially on resource-limited devices like edge devices and CPUs.

Here, we distill a fine-tuned BERT model for text classification. The student model is a smaller BERT variant with fewer parameters, designed for efficiency. You can find its [Huggingface repo here](https://huggingface.co/google/bert_uncased_L-10_H-256_A-4). Google introduced this model alongside 23 other compact BERT-like models to highlight distillation’s impact on natural language processing. After distillation, we will evaluate and compare both models, focusing on speed and accuracy in a CPU environment.

Although our teacher model, with ~100 million parameters, is not as large as state-of-the-art models, this example demonstrates how distillation works. The model was fine-tuned on a Runpod A30 GPU for two epochs, yielding the following results:

- **Epoch**: 2.0  
- **Eval Accuracy**: 0.9462  
- **Step**: 6750  

You can follow along using a free Google Colab environment.

### Dataset Overview
We use the [AG-NEWS News Topic Classification Dataset](https://huggingface.co/datasets/SetFit/ag_news), a subset of the original AG-NEWS dataset. It includes ~120,000 news articles categorized into four labels: world, sports, business, and sci/tech.

### Implementation
We first import dependencies, declare variables, and initialize our tokenizer using Huggingface's `AutoTokenizer` class from the Transformers library.

```bash
pip install transformers datasets torch pandas tqdm
```
```python
dataset_ckpt = 'ag_news'
teacher_model_ckpt = 'odunola/bert-base-uncased-ag-news-finetuned-2' #our already finetuned teacher model
student_model_ckpt = 

from huggingface_hub import notebook_login
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForSequenceClassification
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from time import perf_counter
import pandas as pd


data = load_dataset(dataset_ckpt)
train_test = data['train'].train_test_split(test_size = 0.2)
valid_data = train_test['test']
train_data = train_test['train']
test_data = data['test']

def get_num_rows(dataset):
  return dataset.num_rows

print(f'Train set has {get_num_rows(train_data)} texts')
print(f'Valid set has {get_num_rows(valid_data)} texts')
print(f'Test set has {get_num_rows(test_data)} texts')
```

Next we define a class that inherits from Pytorch's Dataset class and construct our dataset.

```python
#now we would utilise pytorch's Dataset andDataloader classes to create our dataset

class MyData(Dataset):
  def __init__(self, data):
    targets = data['label']
    texts = data['text']

    tokens = tokenizer(texts, return_tensors = 'pt', truncation = True, padding = 'max_length', max_length = 150)
    self.input_ids = tokens['input_ids']
    self.attention_mask = tokens['attention_mask']
    self.targets = torch.tensor(targets)
    self.length = len(texts)
  def __len__(self):
    return self.length
  def __getitem__(self, index):
    return self.input_ids[index], self.attention_mask[index], self.targets[index]


train_data = MyData(train_data)
valid_data = MyData(valid_data)
test_data = MyData(test_data)

# now we build our loaders
batch_size = 64
train_loader = DataLoader(train_data,batch_size = batch_size)
valid_loader = DataLoader(valid_data, batch_size = batch_size)
test_loader = DataLoader(test_data, batch_size = batch_size)
```

Great! Looks like we are progressing well. Now let us write a simple function that helps us calculate accuracy. We should also test the current accuracy and speed performace of our teacher model

```python
#first we install define our device and download our teacher model from huggingface
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_ckpt).to(device)


# we define a function to help us compute accuracy as we train, we would also define another function to measure time ellapsed
def accuracy_score(batch, model):
  with torch.no_grad():
    outputs = model(
        batch[0].to(device),
        batch[1].to(device)
    )
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim = 1)
    class_predictions = torch.argmax(probabilities, dim = 1)
    acc = torch.mean((class_predictions == batch[2].to(device)).to(torch.float)).data.item()
    return acc

#now let us test!

accuracy = 0.0
time_taken = 0.0
count = 0
for batch in tqdm(test_loader):
  start_time = perf_counter()
  score = accuracy_score(batch, teacher_model)
  end_time = perf_counter()
  accuracy += score
  time_taken += end_time - start_time

print('\n\n')
print(f"number of samples in each batch is {len(batch[0])}")
print(f'number of batch is {len(test_loader)}')
print(f"accuracy is {accuracy / len(test_loader):.2f}")
print(f'time taken per batch is {time_taken / len(test_loader):.6f}')
```

![results](artifacts/knowledge/2023-nov.webp)
Now, we download our student model off from huggingface. We also define our number of epochs, learning rate, loss functions and optimizer. We also increase dropout to help prevent overfitting.
We would talk more on the loss function in a bit.

```python
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_ckpt, num_labels = 20).to(device)
student_model.dropout = nn.Dropout(0.3) #Increase dropout to improve generalization.
epochs = 5#we train for5epochs
learning_rate = 2e-5
entropy_loss = nn.CrossEntropyLoss() #cross entropy loss
temperature = 2.0
alpha = 0.5
criterion = nn.KLDivLoss(reduction = 'batchmean') #KL Divergence Loss
optimizer = optim.Adam(student_model.parameters(), lr = learning_rate)
```

Now, let's compare the number of parameters of each of our models

```python
def get_parameter_count(model):
  num_params = sum(p.numel() for p in model.parameters())
  return num_params

print(f'teacher model has {(get_parameter_count(teacher_model)/1000000):.2f} parameters')
print(f'student model has {(get_parameter_count(student_model)/1000000):.2f} parameters')

```
![results](artifacts/knowledge/model_params_comparison.webp)

Distilling knowledge involves training a smaller student model to mimic the outputs of a larger teacher model. The student learns not only from the teacher's correct predictions but also from its probabilistic output distribution. This is achieved by minimizing the differences between their probability distributions during training.

Our objective is to align the student’s probability distribution with the teacher’s. After training, both models should produce similar probabilities and focus on the same data features. To accomplish this, we minimize two key losses: **Cross-Entropy (CE) Loss** for classification tasks and **Kullback-Leibler (KL) Divergence Loss** for aligning distributions.

#### KL Divergence Formula
The KL Divergence loss measures how one probability distribution \( q(x) \) diverges from another \( p(x) \):
\[ 
D_{KL}(p \parallel q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} 
\]
This encourages the student model to replicate the teacher’s distribution, including its uncertainty.

#### Cross-Entropy Formula
For classification tasks, the CE loss compares the predicted probabilities \( \hat{y} \) with the true labels \( y \):
\[
CE(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)
\]
Here, \( CE \) penalizes incorrect predictions, ensuring the student learns to classify accurately.

#### Why KL Divergence?
KL Divergence enables the student model to capture both predictions and uncertainties from the teacher, making it ideal for distillation. Mean Squared Error (MSE) is another option, measuring the squared differences between probabilities. However, MSE enforces an exact match with the teacher’s outputs, which may not be desirable in cases requiring flexibility.

```python
import pandas as pd

# Lists to store training and validation metrics
training_loss_list = []
training_kd_loss_list = []
training_accuracy_list = []
valid_loss_list = []
valid_accuracy_list = []

#starting loop
for epoch in tqdm(range(epochs), total=epochs):
    student_model.train()
    train_loss = 0.0
    train_kd_loss = 0.0
    train_accuracy = 0.0
    valid_loss = 0.0
    valid_accuracy = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        target_tensors = batch[2].to(device)

        # Student model predictions
        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits
        ce_loss = entropy_loss(student_logits, target_tensors).data.item()

        # We extract teacher model logits
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Knowledge distillation loss (KD divergence)
        kd_loss = temperature ** 2 * criterion(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1)
        )

        # Combined loss
        loss = alpha * ce_loss + (1. - alpha) * kd_loss
        loss.backward()
        optimizer.step()

        # Update training metrics
        train_kd_loss += kd_loss.data.item()
        train_loss += loss
        accuracy = accuracy_score(batch, student_model)
        train_accuracy += accuracy

    student_model.eval()
    for batch in valid_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        target_tensors = batch[2].to(device)

        # Validation loss
        output = student_model(input_ids=input_ids, attention_mask=attention_mask)
        val_loss = entropy_loss(output.logits, target_tensors)
        valid_loss += val_loss.data.item()

        # Update validation accuracy
        accuracy = accuracy_score(batch, student_model)
        valid_accuracy += accuracy

    # Calculate average metrics
    train_accuracy /= len(train_loader)
    valid_accuracy /= len(valid_loader)
    train_loss /= len(train_loader)
    train_kd_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    # Append metrics to lists
    training_kd_loss_list.append(train_kd_loss)
    training_loss_list.append(train_loss.cpu().detach().numpy())
    training_accuracy_list.append(train_accuracy)
    valid_loss_list.append(valid_loss)
    valid_accuracy_list.append(valid_accuracy)

    # Print and store metrics
    print(f"""
    After epoch {epoch + 1}:
    Training loss (entropy): {train_loss}
    Kullback-Leibler (KL) divergence loss: {train_kd_loss}
    Validation loss (entropy): {valid_loss}
    Training accuracy: {train_accuracy}
    Validation accuracy: {valid_accuracy}
    """)

# Create a DataFrame to store the metrics
metrics = pd.DataFrame({
    'training_loss': training_loss_list,
    'training_kd_loss': training_kd_loss_list,
    'training_accuracy': training_accuracy_list,
    'valid_loss': valid_loss_list,
    'valid_accuracy': valid_accuracy_list
})
```
# Details of the PyTorch Training Loop

Let’s talk a bit about the details of this PyTorch training loop, which follows a standard structure but incorporates some unique aspects. Here’s a breakdown:

1. **Temperature Parameter**  
   - We introduce a “temperature” parameter in our loss calculations. This parameter scales the logits of both the student and teacher models.  
   - The purpose is to make these logits less extreme, and the temperature value influences the softness of the resulting probability distributions.  
   - A higher temperature leads to softer, more spread-out distributions.  
   - In deep learning, we use the temperature to strike a balance between favoring precise, sharp predictions and giving weight to softer, less confident predictions.  

2. **Alpha Parameter**  
   - We employ an “alpha” parameter to determine the balance between two different losses when calculating the overall training loss.  
   - This parameter controls how much each of these losses contributes to the final combined loss. It’s a critical part of the process.  
   - When alpha approaches 1, it highlights the importance of the cross-entropy loss, emphasizing exact match with target labels.  
   - Conversely, when alpha approaches 0, it emphasizes the KL divergence loss, focusing on capturing the broader knowledge and softer predictions.  
   - In our case, with an alpha set to 0.5, both losses are given equal importance in computing the final loss.  

3. **Overall Loss Calculation**  
   - We consider both the KL divergence loss and the cross-entropy loss when calculating the overall loss during training.  
   - The temperature parameter adjusts the logits, affecting the shape of probability distributions.  
   - The alpha parameter dictates the balance between the two losses, ultimately guiding the model in achieving the desired trade-off between precision and soft knowledge capture.  

After training for 5 epochs, we record the following metrics. Not bad for a model almost 10x smaller than its teacher. We can definitely get better results by using different hyper-parameters or even a better student model. Recall the drastic difference in the number of trainable parameters of both models. You can experiment to get better results with dropout, learning rate, alpha, temperature parameters and also train for more epochs. Keep on the lookout for signs of overfitting though!. We also push our student model and tokenizer to a new huggingface repo.

![training results](artifacts/knowledge/training_results.webp)

```python
student_model.push_to_hub('odunola/google-distilled-ag-news')
tokenizer.push_to_hub('odunola/google-distilled-ag-news')
```
Now, with our test loader we write some code to help us do some bench marks. Take note that we reuse the accuracy_score function of earlier.

```python
accuracy_teacher = 0.0
time_taken_teacher = 0.0

accuracy_student = 0.0
time_taken_student = 0.0
count = 0
for batch in tqdm(test_loader):
  start_time = perf_counter()
  score = accuracy_score(batch, teacher_model)
  end_time = perf_counter()
  accuracy_teacher += score
  time_taken_teacher += end_time - start_time

  start_time = perf_counter()
  score = accuracy_score(batch, student_model)
  end_time = perf_counter()
  accuracy_student += score
  time_taken_student += end_time - start_time


print('\n\n')
print(f"number of samples in each batch is {len(batch[0])}")
print(f'total number of batches is {len(test_loader)}')
print(f"teacher accuracy is {accuracy_teacher / len(test_loader):.2f}")
print(f'time taken per batch for teacher is {time_taken_teacher / len(test_loader):.6f}')
print('\n\n\n')
print(f"student accuracy is {accuracy_student / len(test_loader):.2f}")
print(f'time taken per batch for student is {time_taken_student / len(test_loader):.6f}')
```
![final results](artifacts/knowledge/final_results.webp)

The true evaluation, however, involves conducting the same benchmark in an environment that lacks substantial computing power or GPU acceleration. With the following code I test the performance and speed of both student and teacher model in a free Google Colab CPU-only instance. The subsequent results were documented as follows. (note that we only tested for 4 batches of the test loader because of time,);

```python
teach_model = AutoModelForSequenceClassification.from_pretrained('odunola/bert-base-uncased-ag-news-finetuned-2')
stud_model = AutoModelForSequenceClassification.from_pretrained('odunola/distillbert-distilled-ag-news')
device = 'cpu'

accuracy_teacher = 0.0
time_taken_teacher = 0.0
teacher_model = teach_model.to('cpu')
student_model = stud_model.to('cpu')
accuracy_student = 0.0
time_taken_student = 0.0
count = 1
for batch in tqdm(test_loader):
  start_time = perf_counter()
  score = accuracy_score(batch, teach_model)
  end_time = perf_counter()
  accuracy_teacher += score
  time_taken_teacher += end_time - start_time

  start_time = perf_counter()
  score = accuracy_score(batch, stud_model)
  end_time = perf_counter()
  accuracy_student += score
  time_taken_student += end_time - start_time
  if count == 4:
    break
  count += 1


print('\n\n')
print(f"number of samples in each batch is {len(batch[0])}")
print(f'total number of batches is {len(test_loader)}')
print(f"teacher accuracy is {accuracy_teacher / 4:.2f}")
print(f'time taken per batch for teacher is {time_taken_teacher / 4:.6f}')
print('\n\n\n')
print(f"student accuracy is {accuracy_student / 4:.2f}")
print(f'time taken per batch for student is {time_taken_student / 4:.6f}')
```
![results](artifacts/knowledge/1_xUrGvy4RRYwx7N2z4dXngg.webp)

Please note that the time is in seconds.

In summary, I hope our journey with knowledge distillation has been a smooth one. We’ve cut down on memory usage and significantly sped things up, all while keeping up with our teacher model’s performance. These results speak volumes about how this technique can make things faster and lighter without sacrificing too much quality.
