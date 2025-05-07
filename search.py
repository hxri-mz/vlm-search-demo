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
from transformers import AutoModel, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, AutoProcessor
import torch

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
        model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True).to("cuda")
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
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
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
    
# Configs
output_path = "outputs/"
json_file = "results.json"
output_folder = "output"
# prompt = "Describe the scene with objects and their colors, number of lanes, whether it's marked or not, weather (sunny, rainy, snow, cloudy), and time of day (morning, night, dawn, dusk). Mention pedestrians if present."
prompt = "Describe the road driving scene in detail"

# Streamlit
st.title("VLM based image search")
options = st.radio(
    "Select an option to continue",
    ["Process data", "Search"],
    captions=['', 'Make sure you process data first if not done.',]
)
st.divider()

if options == "Process data":
    opt = st.selectbox(
        "Select a VLM model to process data",
        ("MoonDream", "BLIP", "GIT", "UForm"),
    )
    model, processor = load_model(opt)
    folder_path = st.text_input("Enter image folder path", value="data/")
    if st.button("Process Images"):
        if not os.path.exists(folder_path):
            st.error("Folder path does not exist.")
        else:
            os.makedirs(output_folder, exist_ok=True)
            results = {}

            image_files = [f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            total_images = len(image_files)

            pbar = st.progress(0)
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
                    st.warning(f"Error processing {fname}: {e}")
                    continue
                count += 1
                pbar.progress(count + int(100/total_images))
            with open(output_path+opt+'_'+json_file, "w") as f:
                json.dump(results, f, indent=4)
            pbar.empty()

            st.success(f"Processing complete. Output saved to {output_path+opt+'_'+json_file}")
else:            
    search_term = st.text_input("Enter search term", "")
    opt = st.selectbox(
        "Select a VLM model to use for search",
        ("MoonDream", "BLIP", "GIT", "UForm"),
    )
    if st.button("Search"):
        if search_term:
            if os.path.exists(output_path+opt+'_'+json_file):
                with open(output_path+opt+'_'+json_file, "r") as f:
                    results = json.load(f)

                matches = []
                for filename, data in results.items():
                    description = data.get('description', "")
                    # import pdb; pdb.set_trace()
                    md = data.get('model', "")
                    search_term_lower = search_term.lower()
                    if search_term_lower in description.lower().split():
                        words = description.split()
                        highlighted_words = [
                            # f":orange-badge[**{word}**]" if word.lower() == search_term_lower else word
                            f'<text style="background-color: #f8ff29">{word}</text>' if word.lower() == search_term_lower else word
                            for word in words
                        ]
                        highlighted_description = " ".join(highlighted_words)
                        matches.append({
                            "filename": filename,
                            "description": highlighted_description,
                            "model": md,
                        })
                
                if matches:
                    st.subheader("Matches Found:")
                    for match in matches:                    
                        st.markdown(f"""**File**: {match['filename']} <br> **VLM Model**: {match['model']}""", unsafe_allow_html=True)
                        # st.write(f"**VLM Model**: {match['model']}")
                        st.markdown(body=f"**Description**: {match['description']}", unsafe_allow_html=True, help=None)
                        
                        image_path = os.path.join("data", match['filename'])
                        if os.path.exists(image_path):
                            image = Image.open(image_path)                        
                            st.image(image)
                        else:
                            st.warning(f"Image file {match['filename']} not found.")
                        st.divider()

                else:
                    st.write("No matches found.")
            else:
                st.error(f"No results found. Please process images first.")
        else:
            st.warning(f"Enter a search term.")
