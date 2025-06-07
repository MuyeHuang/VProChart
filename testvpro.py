#!/usr/bin/env python3
import os
from PIL import Image
import torch
from transformers import DonutProcessor
from model.unichart_vision_encoder_decoder import VisionEncoderDecoderModel

# Configuration
os.environ["TRANSFORMERS_OFFLINE"] = "1"
MODEL_PATH = "/path/to/pretrained/model"      # <-- update to your checkpoint directory
IMAGE_DIR = "/path/to/chartqa/images"         # <-- update to your image folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & processor
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
processor = DonutProcessor.from_pretrained(MODEL_PATH)
model.to(device)

# Interactive inference loop
while True:
    img_name = input("input path: ").strip()
    if not img_name.lower().endswith(".png"):
        print("→ Please enter a .png filename")
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.isfile(img_path):
        print(f"→ File not found: {img_path}")
        continue

    question = input("question: ").strip()
    prompt = f"<chartqa> {question} <s_answer>"

    # Tokenize question and prompt
    question_ids = processor.tokenizer(
        question, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # Prepare image
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        query_ids=question_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    answer = processor.batch_decode(outputs.sequences)[0]
    print(answer)
