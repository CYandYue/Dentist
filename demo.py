import openai
import torch
import os
from PIL import Image
import configargparse
from Dentist.model.llava.llava_verifier import LLAVA_Verifier
from transformers import AutoProcessor, LlavaForConditionalGeneration

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config_path', is_config_file=True, default="./Dentist/config/llava/llava_config.ini")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limited_cnt", type=int, default=3)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--openai_key", type=str)
    
    args = parser.parse_args()
    if args.config_path:
        parser = configargparse.ArgParser(default_config_files=[args.config_path])
        parser.add_argument('--config_path', is_config_file=True)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--limited_cnt", type=int, default=3)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--openai_key", type=str)
    
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    
    client = openai.OpenAI(api_key=args.openai_key,base_url="https://api.openai.com/v1")
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    limited_cnt = args.limited_cnt
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, load_in_8bit=True, device_map={"": args.device}, torch_dtype=torch.float16
    )
    
    verifier = LLAVA_Verifier(model, processor, client, device, limited_cnt)
    
    image_path = "./figs/limes.jpg"
    image = Image.open(image_path).convert("RGB")
    
    question = "What's in the picture? How many are there?"
    prompt = "<image>\nUSER: " + question + "\nASSISTANT:"
    
    # to get orriginal LlaVA answer
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=device, dtype=torch.float16)
    generate_ids = model.generate(**inputs, max_new_tokens=200)
    answer_original = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    index = answer_original.find("ASSISTANT:")
    if index != -1:
        answer_original = answer_original[index + len("ASSISTANT:"):]
    
    # to get revised answer  
    answer_revise = verifier.verify_loop(original_image=image, original_q=question, original_a=answer_original)
    
    print(f"original_answer: {answer_original}")
    print(f"revised_answer: {answer_revise}")
    
    
if __name__ == '__main__':
    main()