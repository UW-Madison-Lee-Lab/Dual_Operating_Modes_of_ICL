import openai
from openai.error import RateLimitError
import time

openai_keys = {
    'key': 'yourkey',
}     

def get_final_prompt(prompt, demonstrations, sys_msg):
    demos = '\n\n'.join([f"{demonstration['prompt']}\n\nAnswer: {demonstration['completion']}" for demonstration in demonstrations])
    prompt = f'{sys_msg}{demos}\n\n{prompt}\n\nAnswer: '
    #print(prompt)
    return prompt

def call_gptapi(
    prompt, 
    demonstrations = [], 
    mode = 'chat', 
    model = "gpt-3.5-turbo", 
    openai_key = 'none',
    logprobs = False,
    sys_msg = "You are an assistant. Answer the questions following the example templates.",#"You are a mathematician. Consider the following prime number task and follow the exact instruction.",
):
    
    openai.api_key = openai_keys[openai_key]
    
    if mode == 'chat':
        messages = [{"role": "system", "content": sys_msg}]
        for demonstration in demonstrations:
            messages.append({"role": 'user', 'content': demonstration['prompt']})
            messages.append({"role": 'assistant', 'content': demonstration['completion']})
        messages.append({'role': 'user', 'content': prompt})
            
        success = False
        while not success:
            try:
                #model = 'gpt-4'
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                success = True
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                if isinstance(e, RateLimitError):
                    print('rate limit reached, sleep for 30s')
                    time.sleep(30)
                print(f'ChatGPT {model} request failed. Sending another request...')
                success = False
            
        ans = response.choices[0].message.content.strip()
        
    elif mode == 'completion':
        prompt = get_final_prompt(prompt, demonstrations, sys_msg)
        
        success = False
        while not success:
            try:
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=20,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs = 5,
                )
                success = True
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(e)
                print(f'GPT {model} request failed. Sending another request...')
                success = False
                
        if logprobs:
            ans = response['choices'][0]['logprobs']["top_logprobs"]
        else:
            ans = response['choices'][0]['text'].strip()
        
    else:
        raise KeyError(f'mode {mode} not supported!')

    return ans
