import numpy as np
from pathlib import Path
import shutil
import os
import logging
import sys
sys.path.append('..')

from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
import finbert.utils as tools

#%load_ext autoreload
#%autoreload 2

project_dir = Path.cwd().parent
pd.set_option('max_colwidth', -1)

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--start_seed', type=int, required=True)
parser.add_argument('--end_seed', type=int, required=True)
parser.add_argument('--type', type=str, required=True)

logging.basicConfig(filename='example.log', filemode='w', level=logging.ERROR)


args = parser.parse_args()

def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    test_accuracy = (df[cols[0]] == df[cols[1]]).sum() / df.shape[0]
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    #print("\nClassification Report:")
    #print(classification_report(df[cols[0]], df[cols[1]]))
    return test_accuracy

seeds = [
42,
125380
,160800
,22758
,176060
,193228
,261810
,200160
,23442
,116765
,281304
,221072
,167229
,101871
,233936
,146104
,19360
,174949
,55535
,275520
,54106
,141318
,238732
,73844
,216448
,136371
,200768
,169401
,188578
,99245
,182166
,62493
,219228
,13770
,263283
,24218
,133088
,8580
,301903
,149048
,87890
,290381
,265159
,219974
,69875
,265320
,275826
,154180
,63257
,232949
,224631
,49678
,12824
,54290
,277985
,172183
,203263
,56118
,33255
,255658
,125861
,58743
,40064
,179238
,318164
,287365
,131577
,293134
,10110
,108092
,156481
,202726
,16848
,110176
,201391
,17299
,291232
,51045
,305683
,42988
,243066
,240369
,100539
,7466
,218829
,129766
,113706
,38296
,198126
,218344
,61744
,229938
,94585
,31563
,177343
,148549
,140934
,323847
,182596
,79955
,4381
]

# CUDA_VISIBLE_DEVICES=1 python3 -u run_seeds.py --model_name_or_path /scratch/varunt/finBERT/models/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir 2>&1 --start_seed 0 --end_seed 6 | tee hf_0.001decay_5e-5.txt

#/scratch/varunt/finBERT/models/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir/checkpoint_epoch_6_16750.pt

batch_sizes = [16]
max_seq_lengths = [1024]
learning_rates = [5e-5]
decays = [0.001]
num_epochs = [6]

#print(learning_rates)
#print(decays)

best_validation_accuracies = []
test_accuracies = []
for current_batch_size in batch_sizes:
    for current_max_seq_len in max_seq_lengths:
        for current_learning_rate in learning_rates:
            for current_num_epoch in num_epochs:
                for current_decay in decays:
                    for seed_idx in range(args.start_seed, args.end_seed):
                        try:
                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])

                            lm_path = args.model_name_or_path #"/scratch/venkats/finbert_pretrained_weights" #project_dir/'models'/'language_model'/'finbertTRC2'
                            cl_path = "/scratch/varunt/finbert_clpath/" + args.type + "/" + "seed_" + str(seeds[seed_idx]) + "/" + lm_path + "/"  #project_dir/'models'/'classifier_model'/'finbert-sentiment'
                            cl_data_path = "/scratch/varunt/finBERT/datasets/" #project_dir/'data'/'sentiment_data'mport ipdb; ipdb.set_trace()
                            print ("==========================================================")

                            try:
                                # Clean the cl_path
                                shutil.rmtree(cl_path) 
                            except:
                                pass

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir="./cache",num_labels=3)

                            print("this is a test")
                            
                            print ("> -------------- < ")
                            config = Config(   data_dir=cl_data_path,
                                               bert_model=bertmodel,
                                               num_train_epochs=current_num_epoch,
                                               model_dir=cl_path,
                                               max_seq_length = current_max_seq_len,
                                               train_batch_size = current_batch_size,
                                               learning_rate = current_learning_rate,
                                               output_mode='classification',
                                               warm_up_proportion=0.2,
                                               local_rank=-1,
                                               discriminate=True,
                                               decay=current_decay,
                                               seed = seeds[seed_idx],
                                               gradual_unfreeze=True)
                            print (f"Seed: Count: {seed_idx} (Abs: {seeds[seed_idx]})")
                            print (f"LM Path: {lm_path}")
                            print ("> -------------- < ")

                            from pathlib import Path
                            config.model_dir = Path(config.model_dir)

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            finbert = FinBert(config)
                            finbert.base_model = 'gpt2' #'bert-large-uncased' #'bert-base-uncased'
                            finbert.config.discriminate=False # True
                            finbert.config.gradual_unfreeze=False # True

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            finbert.prepare_model(label_list=['positive','negative','neutral'])

                            # Get the training examples
                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            train_data = finbert.get_data('train')

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            model = finbert.create_the_model()

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            trained_model, best_validation_acc = finbert.train(train_examples = train_data, model = model)
                            best_validation_accuracies.append(best_validation_acc)

                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            test_data = finbert.get_data('test')


                            random.seed(seeds[seed_idx])
                            np.random.seed(seeds[seed_idx])
                            torch.manual_seed(seeds[seed_idx])
                            results = finbert.evaluate(examples=test_data, model=trained_model)
                            results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))
                            print ("Results:")
                            test_accuracy = report(results,cols=['labels','prediction','predictions'])
                            test_accuracies.append(test_accuracy)
                            print ("==========================================================")
                        except RuntimeError as e:
                            if 'out of memory' in str(e) and not raise_oom:
                                print('| WARNING: ran out of memory, retrying batch')
                                continue
                            else:
                                print (e)
                                print ("Hit some random OOM")
                                continue
                    # new seed                    
                    print("*"*40)
                    print("*"*40)
                    print("Final Results:")
                    print("Current Learning Rate: {}, Current Decay: {}".format(current_learning_rate, current_decay))
                    max_validation_accuracy = max(best_validation_accuracies)
                    avg_validation_accuracy = np.mean(best_validation_accuracies)
                    corresponding_index = best_validation_accuracies.index(max_validation_accuracy)
                    corresponding_test_accuracy = test_accuracies[corresponding_index]
                    avg_test_accuracy = np.mean(test_accuracies)
                    stddev_test_accuracy = np.std(test_accuracies)

                    print("Validation Accuracies: {}".format(best_validation_accuracies))
                    print("Test Accuracies: {}".format(test_accuracies))
                    print("Best Validation Accuracy: {}".format(max_validation_accuracy))
                    print("Avg Validation Accuracy: {}".format(avg_validation_accuracy))
                    print("Corresponding Test Accuracy: {}".format(corresponding_test_accuracy))
                    print("Average Test Accuracy: {}".format(avg_test_accuracy))
                    print("Std Dev Test Accuracy: {}".format(stddev_test_accuracy))
                    print("*"*40)
                    print("*"*40)

                    # reset stats per each LR, Decay pair
                    best_validation_accuracies = []
                    test_accuracies = []

                    try:
                        # Clean the cl_path
                        shutil.rmtree(cl_path) 
                    except:
                        pass
