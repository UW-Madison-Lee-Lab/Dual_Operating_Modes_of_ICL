from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
'''
def call_mistral(
    prompt, 
    model = None, 
    tokenizer = None,
    max_output_length = 64,
):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Move input to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # Generate a sequence of text
    output = model.generate(input_ids, max_length=max_output_length, num_return_sequences=1)

    # Decode the output
    generated_text1 = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text2 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text1, generated_text2
'''

def call_mistral(
    prompt, 
    model = None, 
    tokenizer = None,
    max_output_length = 64,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    output = model.generate(**inputs, max_new_tokens=50)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_str, output_str
'''
    # Encode the input with attention mask
    encoding = tokenizer.encode_plus(
        prompt, 
        return_tensors='pt', 
        #max_length=max_output_length,
        #padding='max_length', 
        #truncation=True
    )
    input_ids = encoding['input_ids']
    #print(input_ids)
    attention_mask = encoding['attention_mask']
    #print(attention_mask)
    # Move input to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    # Generate a sequence of text
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=max_output_length, 
        num_return_sequences=1
    )

    # Decode the output
    generated_text1 = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text2 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text1, generated_text2

def call_mistral(
    prompt, 
    model = None, 
    tokenizer = None,
    max_output_length = 64,
):
    sampling_params = SamplingParams(temperature=0, max_tokens=max_output_length, stop=['</s>'])
    output = model.generate(prompt, sampling_params, use_tqdm=False)
    return output[0].outputs[0].text, output[0].outputs[0].text
'''
