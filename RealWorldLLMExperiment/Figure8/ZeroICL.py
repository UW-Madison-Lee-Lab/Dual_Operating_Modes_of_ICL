#from call_openai import call_gptapi
import random
#import tiktoken
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

s = 20  # You can adjust this value as needed

# Set the global font size
plt.rcParams['font.size'] = s


classification_datasets = ['glue-mrpc', 'glue-rte', 'tweet_eval-hate', 'sick', 'poem_sentiment']
multichoice_datasets = ['openbookqa', 'commonsense_qa', 'superglue-copa', 'ai2_arc']

#model = 'gpt4'
#model = 'mistral'
#model = 'mixtral'
#model = 'llama2-13b-hf'
#model = 'llama2-70b-hf'

total_k_list = [0,1,2,4,8,16,32,64,128]
total_kstr_list = [r'$0$']+[r'$2^{%d}$'%i for i in range(8)]

fig, axs = plt.subplots(1, 5, figsize=(24, 3.2), sharey=True)

model_dictionary = {
    'mistral': [0,1,2,4,8,16,32,64,128],
    'llama2-13b-hf': [0,1,2,4,8,16,32],
    'mixtral': [0,1,2,4,8,16,32,64,128],
    'llama2-70b-hf': [0,1,2,4,8,16,32],
    'gpt4': [0,1,2,4,8,16,32,64],
}

model2title = {
    'mistral': 'Mistral 7B',
    'llama2-13b-hf': 'Llama 2 13B',
    'mixtral': 'Mixtral 8x7B',
    'llama2-70b-hf': 'Llama 2 70B',
    'gpt4': 'GPT-4',
    }

for index, (model, k_list) in enumerate(model_dictionary.items()):
    
    save_file_name = 'results/' +model+ '/'
    #data_name = classification_datasets[2]
    
    num_subsample = 1
        
    random_seed_list = [13,21,42,87,100]
    #k_list = [0,1,2,4,8,16,32,64,128]#[::2]
    
    ave_acc_gold = 0
    ave_acc_rand = 0
    for data_name in classification_datasets:
        with open(save_file_name+'D_' +data_name+ '.pickle', 'rb') as handle:
            D = pickle.load(handle)
            #print(D)
            
            acc_gold_list = []
            acc_rand_list = []
            for k in k_list:
                acc_gold_list.append(D[k]['gold_rate'])
                acc_rand_list.append(D[k]['rand_rate'])
            '''
            plt.figure()
            plt.plot(acc_gold_list, label='gold')
            plt.plot(acc_rand_list, label='random')
            '''
            ave_acc_gold += np.array(acc_gold_list)
            ave_acc_rand += np.array(acc_rand_list)
            
            '''
            D[k] = {}
            D[k]['gold'] = gold_correct
            D[k]['rand'] = rand_correct
            D[k]['count'] = count
            D[k]['gold_rate'] = gold_correct/count
            D[k]['rand_rate'] = rand_correct/count
            '''
        #for k in k_list:
        #    #print(k, sum(D[k]['gold_example'])/D[k]['count'])
        #    print(k, D[k]['gold_example'])
    
    ave_acc_gold = 1-ave_acc_gold/5
    ave_acc_rand = 1-ave_acc_rand/5
    
    axs[index].plot(ave_acc_gold, label='w/ true labels', linestyle='solid', color='orange', linewidth=4)
    axs[index].plot(ave_acc_rand, label='w/ random labels', linestyle='dashdot', color='blue', linewidth=4)
    
    #plt.plot(ave_acc_gold, label='gold')
    #plt.plot(ave_acc_rand, label='random')
    axs[index].set_title(model2title[model])
    #plt.title(model)
    
    axs[index].set_xticks(np.arange(len(total_k_list)), total_kstr_list)
#axs[0].legend(loc='upper left', bbox_to_anchor=(-0.04, 1.04))
axs[4].legend(loc='lower left', bbox_to_anchor=(0.00, 0.5))#
axs[2].set_xlabel('Number of In-Context Examples '+r'$(k)$')
axs[0].set_ylabel('Classification Error')
axs[0].set_ylim([1-0.9,1-0.2])
plt.savefig('Figure8.pdf', bbox_inches='tight')


