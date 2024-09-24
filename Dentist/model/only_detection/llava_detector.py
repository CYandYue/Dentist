from ..verifier import Verifier
import torch

class LLAVA_Detector(Verifier):
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