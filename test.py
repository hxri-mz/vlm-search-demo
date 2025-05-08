from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

model_id = "amazon/MistralLite"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             use_flash_attention_2=True,
                                             device_map="auto",)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
prompt = "<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>"

sequences = pipeline(
    prompt,
    max_new_tokens=400,
    do_sample=False,
    return_full_text=False,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"{seq['generated_text']}")
