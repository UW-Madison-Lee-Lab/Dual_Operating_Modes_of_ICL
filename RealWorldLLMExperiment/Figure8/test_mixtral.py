#from call_openai import call_gptapi
from call_mistral import call_mistral
import random
import tiktoken
import json
import pickle
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from vllm import LLM
import time
from tqdm import tqdm

def count_tokens(sentence, tokenizer):
    # Initialize the GPT-2 tokenizer
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the input sentence
    tokens = tokenizer.encode(sentence)

    # Count the number of tokens
    token_count = len(tokens)

    return token_count
    
def samples2prompt(data_name, train_sample_list, test__sample, mode='random',max_input_length=None,constrain=True,tokenizer=None):
    enc = tiktoken.encoding_for_model("gpt-4")
    if max_input_length != None:
        total_length = max_input_length #8192
    else:
        total_length = 999999999999999999
    # train -> context
    
    classification_datasets = ['glue-mrpc', 'glue-rte', 'tweet_eval-hate', 'sick', 'poem_sentiment']
    
    if data_name in ['glue-mrpc', 'glue-rte', 'tweet_eval-hate']:
        test__prompt = 'Input: ' + test__sample['x'] + '\n' + 'Output (directly return ' +test__sample['opts'][0]+ ' or ' +test__sample['opts'][1]+ '): '
    if data_name in ['sick', 'poem_sentiment']:
        test__prompt = 'Input: ' + test__sample['x'] + '\n' + 'Output (directly return ' +test__sample['opts'][0]+ ' or ' +test__sample['opts'][1]+ ' or ' +test__sample['opts'][2]+ '): '
        
    
    if tokenizer == None:
    	total_length -= len(enc.encode(test__prompt))
    else:
    	total_length -= count_tokens(test__prompt, tokenizer)
    
    context = ''
    count = 0
    for train_sample in train_sample_list:
        example_x = 'Input: ' + train_sample['x']
        if mode == 'gold':
            if data_name in ['glue-mrpc', 'glue-rte', 'tweet_eval-hate']:
                example_y = 'Output (directly return ' +train_sample['opts'][0]+ ' or ' +train_sample['opts'][1]+ '): ' + train_sample['y']
            if data_name in ['sick', 'poem_sentiment']:
                example_y = 'Output (directly return ' +train_sample['opts'][0]+ ' or ' +train_sample['opts'][1]+ ' or ' +train_sample['opts'][2]+ '): ' + train_sample['y']
        elif mode == 'random':
            if data_name in ['glue-mrpc', 'glue-rte', 'tweet_eval-hate']:
                example_y = 'Output (directly return ' +train_sample['opts'][0]+ ' or ' +train_sample['opts'][1]+ '): ' + random.choice(train_sample['opts'])
            if data_name in ['sick', 'poem_sentiment']:
                example_y = 'Output (directly return ' +train_sample['opts'][0]+ ' or ' +train_sample['opts'][1]+ ' or ' +train_sample['opts'][2]+ '): ' + random.choice(train_sample['opts'])
        
        train_prompt = example_x + '\n' + example_y + '\n\n'
        #total_length -= len(enc.encode(train_prompt))
        if tokenizer == None:
            total_length -= len(enc.encode(train_prompt))
        else:
            total_length -= count_tokens(train_prompt, tokenizer)
        if total_length >= 0:
            context += train_prompt
            count += 1
        else:
            break
        
    #print('k = '+str(count))
    prompt = context + test__prompt
    
    y = test__sample['y']
    return {'prompt':prompt, 'y':y, 'count':count}


def load_tran_test_samples(data_name, k, random_seed):
    if k > 0:
        train_file_name = 'data/'+ data_name + '/' + data_name +'_' +str(k)+ '_' + str(random_seed)+ '_train.jsonl'
        test__file_name = 'data/'+ data_name + '/' + data_name +'_' +str(k)+ '_' + str(random_seed)+ '_test.jsonl'
    elif k == 0:
        train_file_name = 'data/'+ data_name + '/' + data_name +'_' +str(1)+ '_' + str(random_seed)+ '_train.jsonl'
        test__file_name = 'data/'+ data_name + '/' + data_name +'_' +str(1)+ '_' + str(random_seed)+ '_test.jsonl'
        
    train_sample_list = []
    if k > 0:
        with open(train_file_name, 'r') as file:
            for line in file:
                # Parse each line as a JSON object
                json_obj = json.loads(line)
                #print(json_obj)
                train_sample_list.append(
                    {'x': json_obj['input'],
                     'y': json_obj['output'],
                     'opts': json_obj['options']}
                )
    
    test__sample_list = []
    with open(test__file_name, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line)
            #print(json_obj)
            test__sample_list.append(
                {'x': json_obj['input'],
                 'y': json_obj['output'],
                 'opts': json_obj['options']}
                )
    # subsample
    test__sample_list = random.sample(test__sample_list, num_subsample)
    
    return train_sample_list, test__sample_list


classification_datasets = ['tweet_eval-hate', 'glue-mrpc', 'glue-rte', 'sick', 'poem_sentiment']
multichoice_datasets = ['openbookqa', 'commonsense_qa', 'superglue-copa', 'ai2_arc']

max_input_length = 4096*32
max_output_length = 64

num_subsample = 1
    
random_seed_list = [13,21,42,87,100]
    
k_list = [0,1,2,4,8,16,32,64,128][::-1]



# Initialize the model ID
model_id = "mistralai/Mixtral-8x7B-v0.1"
save_file_name = 'results/mixtral'

tokenizer = None
model = LLM(model=model_id, tensor_parallel_size=8)#, enforce_eager=True, dtype=torch.float32)
'''
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
#model = AutoModelForCausalLM.from_pretrained(model_id)

# Check if a GPU is available and if so, use it
if torch.cuda.is_available():
    model = model.to("cuda")
    print('!!! USE GPU !!!')
else:
    print('!!! NO GPU !!!')
'''
print('!!! START !!!')
start_time = time.time()
#model.config.pad_token_id = model.config.eos_token_id
for index,data_name in enumerate(classification_datasets):
    print(index, data_name)
    D = {}
    for k in k_list: #num_examples    
        # inference
        count = 0
        gold_correct = 0
        rand_correct = 0
        gold_example_num = []
        rand_example_num = []
        for random_seed in random_seed_list:
            train_sample_list, test__sample_list = load_tran_test_samples(data_name, k, random_seed)
            for test__sample in tqdm(test__sample_list):
                
                count += 1
                random.shuffle(train_sample_list)
                #print('############################')
                prompty = samples2prompt(data_name, train_sample_list, test__sample, max_input_length=max_input_length, mode='gold', tokenizer=tokenizer)
                gold_example_num.append(prompty['count'])
                #if k == 1:
                #    print('############################')
                #    print(prompty['prompt'])
                
                #ans = call_gptapi(prompty['prompt'], openai_key = 'zl21', model = 'gpt-4')
                whole, ans = call_mistral(prompty['prompt'], model=model, tokenizer=tokenizer, max_output_length=max_output_length)
                #print('############')
                #print(ans[:20])
                #print('############')
                if prompty['y'] == ans.lower()[:len(prompty['y'])]:
                    gold_correct += 1
                if prompty['y'] == ans.lower()[1:len(prompty['y'])+1]:
                    gold_correct += 1
                
                #print('############################')
                prompty = samples2prompt(data_name, train_sample_list, test__sample, max_input_length=max_input_length, mode='random', tokenizer=tokenizer)
                rand_example_num.append(prompty['count'])
                #print('############################')
                #print(prompty['prompt'])
                
                #ans = call_gptapi(prompty['prompt'], openai_key = 'zl21', model = 'gpt-4')
                whole, ans = call_mistral(prompty['prompt'], model=model, tokenizer=tokenizer, max_output_length=max_output_length)
                #print('############')
                #print(ans[:20])
                #print('############')
                if prompty['y'] == ans.lower()[:len(prompty['y'])]:
                    rand_correct += 1
                if prompty['y'] == ans.lower()[1:len(prompty['y'])+1]:
                    rand_correct += 1 
                
            print(gold_correct, rand_correct, count, int(sum(gold_example_num)/count), int(sum(rand_example_num)/count))
                
        D[k] = {}
        D[k]['gold'] = gold_correct
        D[k]['rand'] = rand_correct
        D[k]['count'] = count
        D[k]['gold_rate'] = gold_correct/count
        D[k]['rand_rate'] = rand_correct/count
        D[k]['gold_example'] = gold_example_num
        D[k]['rand_example'] = rand_example_num
        
        with open(save_file_name+'/D_' +data_name+ '.pickle', 'wb') as handle:
            pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(save_file_name+'/D_' +data_name+ '.pickle', 'rb') as handle:
            D = pickle.load(handle)
            
        print(D)
        end_time = time.time()
        print((end_time-start_time)/60,'mins')
        
