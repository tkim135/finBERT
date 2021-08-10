import numpy as np
import io
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import csv
import sys
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from accelerate import Accelerator

# tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialogRPT-updown')
# model = GPT2ForSequenceClassification.from_pretrained('microsoft/DialogRPT-updown')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs

class FinanceDataset(Dataset):
    r"""
    PyTorch Dataset class for loading data.
    This is where the data parsing happens.
    This class is built with reusability in mind: it can be used as is as.
    Arguments:
    path (:obj:`str`):
        Path to the data partition.
    """

    def __init__(self, data_dir, phase):
        self.texts = []
        self.labels = []
        self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)
        # Number of exmaples.
        self.n_examples = len(self.labels)
    def __len__(self):
        r"""
        When used `len` return the number of examples.
        """
        return self.n_examples
    def __getitem__(self, item):
        r"""
        Given an index return an example from the position.
        Arguments:
        item (:obj:`int`):
            Index position to pick an example to return.
        Returns:
            :obj:`Dict[str, str]`: Dictionary of inputs that contain text and asociated labels.
        """
        return {
            'text':self.texts[item],
            'label':self.labels[item]
        }
    def _read_tsv(self, input_file):
        r"""Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text = line[1]
            label = line[2]
            self.texts.append(text)
            self.labels.append(label)

def train(accelerator, model, dataloader, optimizer, scheduler, device):
    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0
    
    # Put the model into training mode.
    model.train()
    
    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].cpu().numpy().flatten().tolist()
        # move batch to device
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()
        
        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]
        
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
        
        # Perform a backward pass to calculate the gradients.
        #loss.backward()
        accelerator.backward(loss)
        
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        
        # Update the learning rate.
        scheduler.step()
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        
        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)
        
        # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss



def validation(accelerator, model, dataloader, device):
    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0
    
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    
    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        
        # add original labels
        true_labels += batch['labels'].cpu().numpy().flatten().tolist()
        
        # move batch to device
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()
            
            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            
            # update list
            predictions_labels += predict_content
            
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
        
    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def main(lr, wd, seed):    
    # setup accelerator
    accelerator = Accelerator(fp16=True)
    
    accelerator.print(">-*-*-*-*-*-*-<")
    accelerator.print(f"LR: {lr}, WD: {wd}, Seed: {seed}")
    accelerator.print(">-*-*-*-*-*-*-<")
    
    # Set seed for reproducibility.
    set_seed(seed)

    # Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
    epochs = 2

    # Number of batches - depending on the max sequence length and GPU memory.
    # For 512 sequence length batch of 10 works without cuda memory issues.
    # For small sequence length can try batch of 32 or higher.
    batch_size = 4

    # Pad or truncate text sequences to a specific length
    # if `None` it will use maximum sequence of word piece tokens allowed by model.
    max_length = 60

    # Look for gpu to use. Will use `cpu` by default if no gpu found.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device

    # Name of transformers model - will use already pretrained model.
    # Path of transformer model - will load your own model from local disk.
    model_name_or_path = 'gpt2-xl'

    # Dictionary of labels and their id - this will be used to convert.
    # String labels to number ids.
    labels_ids = {'negative': 0, 'neutral': 1, 'positive': 2}

    # How many labels are we using in training.
    # This is used to decide size of classification head.
    n_labels = len(labels_ids)
    
    # Get model configuration.
    accelerator.print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    # Get model's tokenizer.
    accelerator.print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token


    # Get the actual model.
    accelerator.print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    #model.to(device)
    model = model.to(accelerator.device)
    accelerator.print('Model loaded to `%s`'%device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)

    data_dir = "/home/ubuntu/finBERT/datasets"
    train_dataset = FinanceDataset(data_dir=data_dir, phase="train")
    validation_dataset = FinanceDataset(data_dir=data_dir, phase="validation")
    test_dataset = FinanceDataset(data_dir=data_dir, phase="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    
    optimizer = AdamW(model.parameters(),
                  lr = lr, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8, # default is 1e-8.
                  weight_decay = wd
                  )
    
    # prepare everything
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
    
    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}
    
    # Loop through each epoch.
    accelerator.print('Epoch')
    for epoch in tqdm(range(epochs)):
        accelerator.print()
        accelerator.print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(accelerator, model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)
        
        # Get prediction form model on validation data. 
        accelerator.print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(accelerator, model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)
        
        # Print loss and accuracy values to see how training evolves.
        #print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
        accelerator.print()
        accelerator.print(">-"*10)
        accelerator.print(f"train_loss: {train_loss}, train_acc: {train_acc}")
        accelerator.print(f"valid_loss: {val_loss}, valid_acc: {val_acc}")
        accelerator.print(">-"*10)
        accelerator.print()
        
        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # Get the prediction from model on test data (just saving it)
    accelerator.print('Test on batches...')
    test_labels, test_predict, test_loss = validation(accelerator, model, test_dataloader, device)
    test_acc = accuracy_score(test_labels, test_predict)
    
    # return the last validation acc and the corresponding test acc
    results = {
        'test_acc_epochs': all_acc['train_acc']
        'val_acc_epochs': all_acc['val_acc'],
        'val_acc': all_acc['val_acc'][-1],
        'test_acc': test_acc,
    }
    accelerator.print(results)
    return results

if __name__ == "__main__":
    #seeds = [42,125380,160800,22758,176060,193228]
    #learning_rates = [5e-5, 5e-4, 5e-6, 1e-7, 5e-7]
    #decays = [0.001, 0.01, 0.0001, 0.005, 0.0005]

    seeds = [42]
    learning_rates = [5e-5]
    decays = [0.001]
    
    for lr in learning_rates:
        for wd in decays:
            path = f"/home/ubuntu/finBERT/gpt_downstream/config_gridsearch/log_{lr}_{wd}.txt"
            # stat tracking
            max_results = None
            valid_accs = []
            test_accs = []
            for seed in seeds:
                results = main(lr=lr, wd=wd, seed=seed)
                valid_accs.append(results['val_acc'])
                test_accs.append(results['test_acc'])
                if max_results == None or results['val_acc'] > max_results['val_acc']:
                    max_results = results
            # collect stats across seeds
            avg_valid_acc = np.mean(valid_accs)
            avg_test_acc = np.mean(test_accs)
            
            stdev_valid_acc = np.std(valid_accs)
            stdev_test_acc = np.std(test_accs)
            
            max_valid_acc = max_results['val_acc']
            corresponding_test_acc = max_results['test_acc']
            #max_valid_acc = max(valid_accs)
            #corresponding_index = valid_accs.index(max_valid_acc)
            #corresponding_test_acc = test_accs[corresponding_index]

            with open(path, 'w') as f:
                # print results
                print("*"*40, file=f)
                print("*"*40, file=f)
                print("Final Results:", file=f)
                print(f"Current Learning Rate: {lr}, Current Decay: {wd}", file=f)
                # valid
                print(f"Validation Accs: {valid_accs}", file=f)
                print(f"Max Validation Acc: {max_valid_acc}", file=f)
                print(f"Avg Validation Acc: {avg_valid_acc}", file=f)
                print(f"Stdev Validation Acc: {stdev_valid_acc}", file=f)
                # test
                print(f"Test Accs: {test_accs}", file=f)
                print(f"Max Test Acc: {corresponding_test_acc}", file=f) # corresponding test accuracy to max validation_acc
                print(f"Avg Test Acc: {avg_test_acc}", file=f)
                print(f"Stdev Test Acc: {stdev_test_acc}", file=f)