from ..verifier import Verifier
import numpy as np
from PIL import Image

class Instructblip_Verifier(Verifier):
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