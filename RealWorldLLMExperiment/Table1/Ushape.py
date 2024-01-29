import random
import numpy as np
import copy
from call_openai import call_gptapi

import time

def number2prompt(integers):
    # integers: [[a,b], [a,b], [a,b]]
    #primes     = data['primes'    ]
    #non_primes = data['non_primes']
    #prompt = 'Given a sequence of numbers, output a binary string where 1 represents a prime number and 0 represents a non-prime number. For the sequence ['
    start = 'You are given examples. Each example has two integer as input and one integer as output. Please provide an answer for the last problems in the math exercise:\n'
    context = ''
    for a,b in integers[:-1]:
        context = context+str(a)+'(?)'+str(b)+'='+str(a+b+1)+'\n'
    context = context+str(integers[-1][0])+'(?)'+str(integers[-1][1])+'='+'\n'
    question = 'Provide your answer directly.'
    
    prompt = start + context + question
    
    return {'prompt':prompt, 'sum':prompt}

def get_llm_results(data, openai_key, model):
    prompt = number2prompt(data)
    print(prompt['prompt'])
    ans1 = call_gptapi(prompt['prompt'], openai_key = openai_key, model=model)
    print(ans1)
    #ans1 = ans1[-5000:]
    #print(ans1)
    #history = [{'prompt': prompt['cot'],
    #           'completion': ans1}]
    
    #ans2 = None#call_gptapi(prompt['sum'], openai_key = openai_key, demonstrations=history, model=model)
    
    #prompt = number2prompt(data, single=True)
    #print(prompt['cot'])
    #ans3 = call_gptapi(prompt['cot'], openai_key = openai_key, model=model)
    #print(ans3)
    return ans1


if 0: #test basic accuracy:
    # 7529 8100 0.9295061728395062
    correct = 0
    count = 0
    for a in np.arange(99,9,-1):
        for b in np.arange(99,9,-1):
            #print(integers)
            integers = [[a,b]]
            preds = get_llm_results(integers, openai_key = 'key', model = 'gpt-4')
            count += 1
            
            preds.replace(' ', '')
            # Split the string by comma
            answers = preds.split(',')
            if int(answers[0]) == (a*b):
                correct += 1
            print(correct,count,correct/count)

if 1: #test coded accuracy:
    # 7529 8100 0.9295061728395062
    sequences = []
    
    pairs = []
    for a in np.arange(99,9,-1):
        for b in np.arange(99,9,-1):
            #if a > b:
            if 1:
                pairs.append([a,b])
            #else:
            #    pairs.append([b,a])
    random.shuffle(pairs)
    
    K = 1
    N = int(np.ceil(len(pairs)/K))
    correct1 = 0
    correct2 = 0
    count = 0
    for i in range(N):
        integers = pairs[i*K:(i+1)*K]
        
        preds = get_llm_results(integers, openai_key = 'key', model = 'gpt-4')
        len_integers = len(integers)
        count += 1
        
        preds = preds.replace(' ', '')
        answers = preds#.split(',')
        #len_answers = len(answers)
        
        
        try:
            if integers[-1][0]+integers[-1][1] == int(answers):
                    correct1 += 1
            else:
                print('**************************************')
        except:
            print('**************************************')
        
        try:
            if integers[-1][0]+integers[-1][1] == (int(answers)-1):
                    correct2 += 1
            else:
                print('**************************************')
        except:
            print('**************************************')
            
        print(count,correct1/count,correct2/count)
