
import re
from PIL import Image
import numpy as np
import openai

debug_flag = False


#------------------------class------------------------------------------------------
# Base class
class Verifier:
    def __init__(self, model, processor, client, device, limited_cnt=3):
        self.model = model
        self.processor = processor
        self.client = client
        self.device = device
        self.limited_cnt = limited_cnt
        
    def verify_loop(self, original_image, original_q, original_a):
        revised_answer = self.verify(original_image, original_q, original_a)
        init = revised_answer
        # Add more examples for better results
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
        
        count = 1
        while count < self.limited_cnt:
            temp = self.verify(original_image, original_q, revised_answer)
            
            input = prompt + "\nPassage 1:{}\nPassage 2:{}".format(revised_answer, temp)
            completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])
            judgement = completion.choices[0].message.content.lower()
            
            # purify
            if "yes" in judgement:
                revised_answer = temp
            elif "no" in judgement:
                return revised_answer
            else:
                return revised_answer
            count += 1
            
        return init
            
            
    
    def verify(self, original_image, original_q, original_a):
        # 1.classify
        # Add more examples for better results
        prompt = '''Role:
        You are now one of my problem classification assistants. 
        Please help me classify the problem into two categories: perception or reasoning(binary classification). 

        Rules:
        1.The classification result is only "perception" or "reasoning". Choose one of the two to output.
        2.If the question focuses on perception ability, answer "perception"; if the question focuses on logical reasoning ability, answer "reasoning"
        3.Don't answer anything else, your answer can only contain "perception" or "reasoning".
        
        Examples:
        1.my input:"how many people are there in this picture?"
        your answer:"perception"
        
        2.my input:"the person in the picture may do what soon?"
        your answer:"reasoning"
        
        3.my input:"There are several options:A. Do the insides of white boxes get hotter than the insides of black boxes when the boxes are left in the sun? B. Do the temperatures inside boxes depend on the sizes of the boxes? Identify the question that Jeanette's experiment can best answer."
        your answer:"reasoning"
        
        4.my input:"Write a detailed description."
        your answer:"perception"
        
        5.my input:"Based on this image, please predict what will happen next?"
        your answer:"reasoning"
        
        Please don't answer anything else, your answer can only contain "perception" or "reasoning"!
        Now please classify the following question according to the example and then answer "perception" or "reasoning":
        '''
        input = prompt + "\nmy input:" + original_q +"\nyour answer:"
        completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])
        classification = completion.choices[0].message.content
        classification = classification.lower()
        
        # 2.purify
        if "perception" in classification:
            classification = "perception"
        elif "reasoning" in classification:
            classification = "reasoning"
        else:
            UserWarning("neither perception or reasoning...!")
            classification = "perception"
        
        # 3.divide-and-conquer treatment
        answer = ""
        if classification == "perception":
            answer =  self.verify_perception(original_image, original_q, original_a)
        elif classification == "reasoning":
            answer =  self.verify_reasoning(original_image, original_q)
        return answer
            
            
                     
    def verify_perception(self, original_image, original_q, original_a):
        # Add more examples for better results
        prompt = '''Role:
        You are my language assistant for generating sub-questions. 
        Please generate sub-questions to verify the caption of the picture based on QA-examples below.
                
        Rules:
        1.The number of sub-questions cannot exceed three.
        2.Extract keywords such as objects, quantities, and locations to generate sub-questions.
        3.Each sub-question should have a different focus.
        4.Don't ask repeated questions in different sub-questions.
        5.If my input contains multiple choice questions, please generate sub-questions based on the question, options and answers.
                
        Examples:
        1.my input:
        "Question: how many people are there in this picture? 
        Answer: there are three people"
        sub-questions you generated:
        "1.Are there three people in this picture?"
        
        2.my input:
        "Question: Write a detailed description for this picture.  
        Answer: The picture shows a man standing on the back of the yellow taxi,with a yellow shirt and black pants, and a blue backpack on his back.The taxi is driving on a city street with cars and taxis in the background."
        sub-questions you generated:
        "1.is there a man standing on the back of a taxi in this picture?
         2.What color are the T-shirt and pants that man wear?
         3.What's in the background?"
                
        3.my input:
        "This organism is a frilled lizard. It is also called Chlamydosaurus kingii. There are several options:
        A. Chlamydosaurus kingii
        B. frilled lizard
        Which is this organism's common name?
        Answer:B. frilled lizard"
        sub-questions you generated:
        "1.What are the characteristics of fried lizard? Do the objects in the picture have these characteristics?
         2.Is there a fried lizard in the picture?
         3.Does the object in the picture look more like a fried lizard than Chlamydosaurus kingii?
        "

        Now please generate verification sub-questions based on my input below.
        my input:
        '''
        my_input = "Question: " + original_q + "\nAnswer: " + original_a + "\nsub-questions you generated:"
        input = prompt + my_input
        completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])
        sub_verification_q = completion.choices[0].message.content
        if debug_flag:
            print("original_sub_questions:\n{}\n".format(sub_verification_q))
        
        # slice
        numbers = re.findall(r'\d+', sub_verification_q)
        splices = re.split(r'\d+', sub_verification_q)
        sub_verification_q=[]
        for i in range(len(numbers)):
            sub_verification_q.append((splices[i+1]).replace("\n", "").replace(". ", ""))
        # print(sub_verification_q)
        
        # vqa
        result = self.vqa_model_evaluatioin(original_image, sub_verification_q)
        if debug_flag:
            print("verified_result:\n{}\n".format(result))
        
        # aggregation
        # Add more examples for better results
        prompt = '''
        Role:
        You are my language assistant for correcting or remaining my passage. 
        Below is a passage and some Q&A pairs. You need to modify the passage or just keep it unchanged based on the Q&A pairs. 
                
        Rules:
        1.The information provided by the Q&A pairs is the ground truth, and the information in the passage may contain errors.
        2.If the passage conflict with the Q&A pairs, find them and correct the passage based on the Q&A pairs. Try to make minimal changes to retain the original sentence. Then give me the passage which have been corrected by you.
        3.If the passage has no confliction with the Q&A pairs, just keep the original passage and give me that.
        4.At any time your output should only be a passage.
                
        Examples:
        1.
        Passage:"There are two apples in the picture, they look stale."
        Q&A pairs:
        Q:How many apples are there in the picture? A:There are three apples in the picture.
        Q:Do these apples look fresh in the picture? A:No, they look stale.
        Your output:"There are three apples in the picture, they look stale."
                
        2.
        Passage:"Man holding a bouquet of flowers in hand for woman"
        Q&A pairs:
        Q:How many people are there in the picture? What are their genders? A:A man and an woman.
        Q:What is the man holding in his hand? A:a bouquet of flowers.
        Your output:"Man holding a bouquet of flowers in hand for woman"

        Now I give you the passage and some Q&A pairs, please follow the examples and give me your output.
        '''
        input = "Passage:{}\n".format(original_a)
        input = input + "QA pairs:\n"
        for item in result:
            input =input + "Q:{} A:{}\n".format(item["question"], item["answer"])
        input = prompt + input + "Your output:"
        completion = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])
        answer_verified = completion.choices[0].message.content
        if debug_flag:
            print("verified_answer:\n{}\n".format(answer_verified))
        return answer_verified
            
        
    # according to the inherited model class, rewrite the following two functions   
    def verify_reasoning(self, original_image, original_q):
        if isinstance(original_image, (Image.Image, np.ndarray)):
            image = self.processor["eval"](original_image).unsqueeze(0).to(self.device)
        else:
            image = original_image
        prompt = "Let's think step by step."
        question = prompt + original_q
        cot_answer = self.model.generate({"image": image, "prompt": question})[0]
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















