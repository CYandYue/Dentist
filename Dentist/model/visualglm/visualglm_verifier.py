from ..verifier import Verifier

class Visualglm_Verifier(Verifier):
    def verify_reasoning(self, original_image_path, original_q):
        base_prompt = "describe this picture"
        _, history = self.model.chat(self.processor, original_image_path, base_prompt, history=[])

        prompt = "Let's think step by step."
        question = prompt + original_q
        cot_answer, _ = self.model.chat(self.processor, original_image_path, question, history=history)
        # print("reasoning_answer:\n{}\n".format(cot_answer))
        return cot_answer
    
    def vqa_model_evaluatioin(self, original_image_path, questions):
        base_prompt = "describe this picture"
        _, history = self.model.chat(self.processor, original_image_path, base_prompt, history=[])
        result = []
        for question in questions:
            answer, _ = self.model.chat(self.processor, original_image_path, question, history=history)
            result.append({"question":question, "answer":answer})
        return result