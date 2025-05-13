'''
VLM based image search and retrival capability demo
---------------------------------------------------
Avaialble models
    -   MoonDream2
    -   Blip2
'''

# Imports
import streamlit as st
import os
import json
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Model loading
@st.cache_resource
def load_model(opt):
    if opt == 'MoonDream':
        model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True,
                device_map={"": "cuda"}
            )
        return model, None
    elif opt == 'BLIP':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
        return model, processor
    elif opt == 'GIT':
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to("cuda")
        return model, processor
    elif opt == 'UForm':
        model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True).to("cpu")
        processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
        return model, processor

def process_data(opt, image, model, processor, prompt):
    if opt == 'MoonDream':
        encoded_image = model.encode_image(image)
        description = model.query(encoded_image, prompt)['answer']
        return description
    
    elif opt == 'BLIP':
        inputs = processor(image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        return description
    
    elif opt == 'GIT':
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption
    
    elif opt == 'UForm':
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cpu")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=256,
                eos_token_id=151645,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text
    
def load_llm_search(lm):
    llmmodel = AutoModelForCausalLM.from_pretrained(
        lm,
        torch_dtype="auto",
        device_map="cuda"
    )
    llmtokenizer = AutoTokenizer.from_pretrained(lm)
    return llmmodel, llmtokenizer

def remove_duplicates(input_list):
    seen = set()
    new_list = []
    for item in input_list:
        if item not in seen:
            new_list.append(item)
            seen.add(item)
    return new_list

def llm_search(terms, words, model, tokenizer):
    prompt = f"'{words}'. Does the paragraph explicitly say anything about atleast one of {terms}?. It has to be a word match either direct, adjective, verb or derived. Reply with True or False."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    
# Configs
output_path = "outputs/"
json_file = "results.json"
output_folder = "outputs"
search_base = 'simple' #'llm'
llm = "Qwen/Qwen2-0.5B-Instruct"
prompt = "Describe the scene with objects and their colors, number of lanes, whether it's marked or not, weather (sunny, rainy, snow, cloudy), and time of day (morning, night, dawn, dusk). Mention pedestrians if present."
# prompt = "Describe the road driving scene in detail"

# Streamlit
st.title("VLM based image search")
st.markdown('Make sure to run  <text style="border: 1.5px solid #72b3b5; border-radius: 8px; padding: 1.5px 6px 3.5px 6px; margin: 0px 6px 0px 6px;">Process data</text>  before searching', unsafe_allow_html=True, help=None)

tab1, tab2 = st.tabs(["Process data", "Search"])

opt = tab1.selectbox(
    "Select a VLM model to process data",
    ("MoonDream", "BLIP", "GIT", "UForm"),
)
model, processor = load_model(opt)
# llmmodel, llmtokenizer = load_llm_search(llm)
folder_path = tab1.text_input("Enter image folder path", value="data/")
if tab1.button("Process Images"):
    if not os.path.exists(folder_path):
        tab1.error("Folder path does not exist.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        results = {}

        image_files = [f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        total_images = len(image_files)

        pbar = tab1.progress(0)
        count = 0
        for idx, fname in enumerate(image_files):
            print(f'processing {idx}')
            image_path = os.path.join(folder_path, fname)
            image = Image.open(image_path).convert("RGB")
            
            try:
                description = process_data(opt, image, model, processor, prompt)
                results[fname] = {
                    "description": description,
                    "model": opt,
                }

            except Exception as e:
                tab1.warning(f"Error processing {fname}: {e}")
                continue
            count += 1
            pbar.progress(count + int(100/total_images))
        with open(output_path+opt+'_'+json_file, "w") as f:
            json.dump(results, f, indent=4)
        pbar.empty()

        tab1.success(f"Processing complete. Output saved to {output_path+opt+'_'+json_file}")

opt = tab2.selectbox(
    "Select a VLM model to use for search",
    ("MoonDream", "BLIP", "GIT", "UForm", "Qwen"),
)

search_term = tab2.text_input("Enter search term", "")

result_thres = tab2.slider(
    "How many results needed?",
    value=2,
)

night = {'tag':'night', 'toggle': False}
morning = {'tag':'morning', 'toggle': False}
sunny = {'tag':'sunny', 'toggle': False}
rainy = {'tag':'rainy', 'toggle': False}
snow = {'tag':'snow', 'toggle': False}
fog = {'tag':'fog', 'toggle': False}

# tab2.write("Select time of day")
# c = tab2.container()
# with tab2:
#     col11, col12, _, _, _, _ = st.columns(6)
#     with col11:
#         night['toggle'] = st.checkbox("Night")
#     with col12:
#         morning['toggle'] = st.checkbox("Morning")

# tab2.write("Select weather")
# with tab2:
#     col21, col22, col23, col24, _, _  = st.columns(6)
#     with col21:
#         sunny['toggle'] = st.checkbox("Clear")
#     with col22:
#         rainy['toggle'] = st.checkbox("Rainy")
#     with col23:
#         snow['toggle'] = st.checkbox("Snow")
#     with col24:
#         fog['toggle'] = st.checkbox("Fog")

if tab2.button("Search"):
    if search_term:
        if os.path.exists(output_path+opt+'_'+json_file):
            with open(output_path+opt+'_'+json_file, "r") as f:
                results = json.load(f)

            matches = []
            for filename, data in results.items():
                description = data.get('description', "")
                md = data.get('model', "")
                search_term_lower = search_term.lower().split()
                words = description.lower().replace(',', '').replace('.', '').split()
                search_terms = []
                for tag in [night, morning, sunny, rainy, snow, fog]:
                    if tag['toggle']:
                        search_terms.append(tag['tag'])
                search_terms=search_terms+search_term_lower
                search_terms = [word for word in search_terms if word not in ['a', 'in', 'on']]
                if search_base == 'llm':
                    resp = llm_search(search_terms, description, llmmodel, llmtokenizer)
                    if resp.lower() == 'true':
                        highlighted_words = [
                                # f":orange-badge[**{word}**]" if word.lower() == search_term_lower else word
                                f'<text style="background-color: #f8ff29">{word}</text>' if word.lower() in search_terms else word 
                                for word in words
                            ]
                        highlighted_description = " ".join(highlighted_words)
                        matches.append({
                            "filename": filename,
                            "description": highlighted_description,
                            "model": md,
                        })
                else:
                    # if set(search_terms).issubset(words):
                    if [i for i in search_terms if i in words]:
                        count = sum(1 for i in search_terms if i in words)
                        highlighted_words = []
                        matched_words = []
                        # highlighted_words = [
                        #     # f":orange-badge[**{word}**]" if word.lower() == search_term_lower else word
                        #     f'<text style="background-color: #f8ff29">{word}</text>' if word.lower() in search_terms else word 
                        #     for word in words
                        # ]
                        for word in words:
                            if word.lower() in search_terms:
                                highlighted_words.append(f'<text style="background-color: #f8ff29">{word}</text>')
                                matched_words.append(word)
                            else:
                                highlighted_words.append(word)
                        highlighted_description = " ".join(highlighted_words)
                        matches.append({
                            "filename": filename,
                            "description": highlighted_description,
                            "model": md,
                            "count": count,
                            "matched_words": matched_words,
                        })
                    
            if matches:
                tab2.subheader("Matches Found:")
                matches.sort(key=lambda x: x["count"], reverse=True)
                matches = matches[:result_thres]#[matches[0]] # select the best
                for match in matches:                    
                    tab2.markdown(f"""**File**: {match['filename']} <br> **VLM Model**: {match['model']} <br> **Matches**: {", ".join(remove_duplicates(match['matched_words']))}""", unsafe_allow_html=True)
                    # tab2.markdown(body=f"**Description**: {match['description']}", unsafe_allow_html=True, help=None)
                    
                    image_path = os.path.join("data", match['filename'])
                    if os.path.exists(image_path):
                        image = Image.open(image_path)                        
                        tab2.image(image)
                    else:
                        tab2.warning(f"Image file {match['filename']} not found.")
                    tab2.divider()

            else:
                tab2.write("No matches found.")
        else:
            tab2.error(f"No results found. Please process images first.")
    else:
        tab2.warning(f"Enter a search term.")
