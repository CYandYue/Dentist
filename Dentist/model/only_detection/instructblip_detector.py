from ..verifier import Verifier
import torch
import numpy as np
from PIL import Image

class Instructblip_Detector(Verifier):
    def detect(self, original_image, original_q, original_a):
        revised_answer = self.verify(original_image, original_q, original_a)
        prompt = '''
                Role:
                You are my language assistant for determining whether there is a conflict between two passages. 
                
                Rules:
                1.Below are two passages. 
                2.If there is conflicting content between the two passages, you should answer "yes"
                3.If there is no conflicting content between the two passages, you should answer "no"
                4.At any time You can only answer yes or no.
                
                Examples:
                1.
                Passage 1: "There are two apples in the picture, they look stale."
                Passage 2: "There are three apples in the picture, they look stale."
                Your answer: "yes"
                
                2.
                Passage 1: "Man holding a bouquet of flowers in hand for woman"
                Passage 2: "There is a man and a woman. The Man holding a bouquet of flowers, maybe he wants to give them to the woman."
                Your answer: "no"
                
                Now I give you two passages, please follow the examples and give me your answer about whether there is a conflict between two passages.
                '''
        
            
        input = prompt + "\nPassage 1:{}\nPassage 2:{}".format(revised_answer, original_a)
        completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])
        judgement = completion.choices[0].message.content.lower()
            
        # purify
        if "yes" in judgement:
            return ""
        elif "no" in judgement:
            return original_a
        else:
            return ""

    def verify_reasoning(self, original_image, original_q):
        if isinstance(original_image, (Image.Image, np.ndarray)):
            image = self.processor["eval"](original_image).unsqueeze(0).to(self.device)
        else:
            image = original_image
        prompt = "Let's think step by step."
        question = prompt + original_q
        cot_answer = self.model.generate({"image": image, "prompt": question})[0]
        # print("reasoning_answer:\n{}\n".format(cot_answer))
        return cot_answer
    
    def vqa_model_evaluatioin(self, original_image, questions):
        if isinstance(original_image, (Image.Image, np.ndarray)):
            image = self.processor["eval"](original_image).unsqueeze(0).to(self.device)
        else:
            image = original_image
        result = []
        for question in questions:
            answer = self.model.generate({"image": image, "prompt": question})[0]
            result.append({"question":question, "answer":answer})
        return result






