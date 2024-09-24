from ..verifier import Verifier
import torch

class LLAVA_Verifier(Verifier):
    def verify_reasoning(self, original_image, original_q):
        question = "<image>\nUSER: Let's think step by step." + original_q + "\nASSISTANT:"
        
        inputs = self.processor(images=original_image, text=question, return_tensors="pt").to(device=self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_length=400)
        cot_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("reasoning_answer:\n{}\n".format(cot_answer))
        index = cot_answer.find("ASSISTANT:")
        if index != -1:
            cot_answer = cot_answer[index + len("ASSISTANT:"):]
        return cot_answer
    
    def vqa_model_evaluatioin(self, original_image, questions):
        result = []
        for question in questions:
            prompt = "<image>\nUSER: " + question + "\nASSISTANT:"
            inputs = self.processor(images=original_image, text=prompt, return_tensors="pt").to(device=self.device, dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=400)
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            index = answer.find("ASSISTANT:")
            if index != -1:
                answer = answer[index + len("ASSISTANT:"):]
            result.append({"question":question, "answer":answer})
        return result