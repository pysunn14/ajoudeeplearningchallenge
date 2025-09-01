import os
import json
import pickle
import hashlib
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image
import io
import base64
from tqdm import tqdm
import argparse

# 설정 
FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
TRAIN_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset.parquet'
TEST_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset_test.parquet'
SAMPLE_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset_sample.parquet'

PREPROCESSED_DATA_PATH = FILE_PATH + 'preprocessed_data'
IMAGE_CACHE_DIR = os.path.join(FILE_PATH, "image_cache")
DOWNLOAD_REPORT_PATH = os.path.join(FILE_PATH, "download_report")

# 추론용 TASK별 프롬프트
def prompt_one_shot(task):

    base_system = '''<SYSTEM> You are a helpful multimodal assistant. Analyze the user's query to identify the specific <USER TASK> and generate a response that strictly follows the format shown in the corresponding example below.
<OUTPUT_RULE> IMPORTANT: You MUST NOT include the <Response> tag.
'''
    
    task_specific_prompts = {
        'summarization': '''<TASK> summarization
<Instruction> Provide a high-level summary of the document's primary function and purpose. Focus on what the document establishes, requires, and authorizes, except procedural details.
<Example>
    <Input> [A legal text about restructuring the federal budget.]
    <Response> Federal Budget Structure Act of 1993 - Amends Federal law to require that the budget the President submits to the Congress be a unified budget comprising an operating budget and a capital budget, each presented separately for total funds, Federal funds, and trust funds.  Restricts the capital budget to the major activities, projects, and programs supporting the acquisition, construction, alteration, and rehabilitation of capital assets.  Includes all other items in the operating budget. 
Requires the following reports to specified congressional committees on capital activities and operating activities associated with:  (1) roadways and bridges, airports and airway facilities, and mass transportation systems; (2) waste water treatment and related facilities; (3) water resource projects; and (4) public buildings.
</SYSTEM>''',

        'math_reasoning': '''<TASK> math_reasoning
<Instruction> Solve the math problem by your reasoning step-by-step. You must show all reasoning steps. 
<Format> Provide your all reasoning steps and final answer with "####" (4 hash) followed by the numerical result.
<Example> 
    <Input> A grocery store sells apples for $2 each and bananas for $1 each. If I buy 3 apples and 5 bananas, how much do I spend in total?
    <Response> 
    Reasoning:
    I need to calculate the total cost of the apples and the bananas. 
    Cost of apples: 3 * 2 = <<3*2=6>> 
    Cost of bananas: 5 * 1 = <<5*1=5>>
    Total cost: 6 + 5 = <<6+5=11>>
    #### 11  
</SYSTEM>''',

        'captioning': '''<TASK> captioning
<Instruction> The goal is to generate a comprehensive and descriptive caption for the image. Your description should be as long and detailed as possible, painting a complete picture of the scene. Consider the following elements to create a rich narrative:

1.  **Main Subjects:** Identify all key subjects in the image, including people, animals, and prominent objects.
2.  **Attributes:** Describe the specific attributes of these subjects. This includes their colors, textures, clothing, and any unique features.
3.  **Actions & Interactions:** Detail the actions of the subjects and how they are interacting with each other or with the environment. What are they doing, and what is the relationship between them?
4.  **Setting & Environment:** Provide a thorough description of the background and setting. Mention the location (e.g., indoors, outdoors, city, nature), lighting, time of day, and any surrounding objects or elements that contribute to the atmosphere.
5.  **Composition & Style:** Analyze the overall composition. Describe the perspective (e.g., close-up, wide shot), the visual style (e.g., a photograph, painting, cartoon), and the overall mood or feeling of the image.

Your final output should be a single, detailed paragraph that weaves all these elements together into a cohesive and comprehensive description.

<Example> 
    [Image: A picture of a golden retriever playing in a park]
    <Response> The image is the cover of a book titled "Devil's Highway" by Hank Janson. The cover features a man in a blue suit and a woman in a pink dress. The man is holding a gun and the woman is looking up at him with a surprised expression on her face. The title of the book is written in red and yellow letters at the top of the cover, with the subtitle "Nine Million Sale" written in smaller letters below. The number "26" is written on the bottom right corner. The background is a dark green color.
</SYSTEM>''',

        'vqa': '''<TASK> vqa
<Instruction> Answer the question about the image in a single word or a very short phrase.
<Format> single word / very short phrase, not long sentence. Don't attach 'Response' keyword.
<Example>
    [Image: A picture of three red apples on a table]
    <Question> How many apples are there?
    <Response> Three
</SYSTEM>''',

        'text_qa': '''<TASK> text-qa  
<Instruction> Based on the provided input, answer the raw text, separate them with a comma (,)
each answer should be single word / very short phrase from the input
<Important> DO NOT repeat the questions in your output!! 
<Format> answer1, answer2, answer3 ...
<Example>   
    <Input> The patient showed a high fever of 39 degrees Celsius. The doctor prescribed medicine. After taking the medicine, his condition improved.
    <Question> ["What was the patient's temperature?", "Did the medicine work?"]
    <Response> 39 degrees Celsius, yes
</SYSTEM>'''
    }
    
    task_prompt = task_specific_prompts.get(task)
    return base_system + task_prompt
    
def prompt_two_shot(task):
    """태스크별 시스템 프롬프트 반환"""
    base_system = '''<SYSTEM> You are a helpful multimodal assistant. Analyze the user's query to identify the specific <USER TASK> and generate a response that strictly follows the format shown in the corresponding example below.
<OUTPUT_RULE> IMPORTANT: You MUST NOT include the <Response> tag.
'''
    
    task_specific_prompts = {
        'summarization': '''<TASK> summarization
<Instruction> Provide a high-level summary of the document's primary function and purpose. Focus on what the document establishes, requires, and authorizes, except procedural details.
<Example 1>
    <Input> [A legal text about restructuring the federal budget.]
    <Response> Federal Budget Structure Act of 1993 - Amends Federal law to require that the budget the President submits to the Congress be a unified budget comprising an operating budget and a capital budget, each presented separately for total funds, Federal funds, and trust funds.  Restricts the capital budget to the major activities, projects, and programs supporting the acquisition, construction, alteration, and rehabilitation of capital assets.  Includes all other items in the operating budget. 
Requires the following reports to specified congressional committees on capital activities and operating activities associated with:  (1) roadways and bridges, airports and airway facilities, and mass transportation systems; (2) waste water treatment and related facilities; (3) water resource projects; and (4) public buildings.

<Example 2>
    <Input> [A legal text amending the National Flood Insurance Act to adjust the phase-in period for premium rate increases.] 
    <Response> Saving Homeowners from Onerous Rate Escalations Act of 2013 or SHORE Act of 2013 - Amends the National Flood Insurance Act of 1968 to direct the Administrator of the Federal Emergency Management Agency (FEMA) to phase in, over an eight-year period, any increase in the flood insurance risk premium rate caused by the prohibition against extending subsidies to new or lapsed policies.  Extends from a 5-year to a 10-year period the phase-in period for premium adjustment increases in the flood insurance risk rate. Prescribes a phase-in rate of: (1) 5% for each of the first 5 years after the effective date of an update, and 15% for each of the 5 ensuing years; and (2) 5% for each of the first 5 years following the effective date of designation as a special flood area of any area not previously so designated, and 15% for each of the 5 ensuing years.
    
</SYSTEM>''',

        'math_reasoning': '''<TASK> math_reasoning
<Instruction> Solve the math problem by your reasoning step-by-step. You must show all reasoning steps. 
<Format> Provide your all reasoning steps and final answer with "####" (4 hash) followed by the numerical result.
<Example 1> 
    <Input> A grocery store sells apples for $2 each and bananas for $1 each. If I buy 3 apples and 5 bananas, how much do I spend in total?
    <Response> 
    I need to calculate the total cost of the apples and the bananas. 
    Cost of apples: 3 * 2 = <<3*2=6>> 
    Cost of bananas: 5 * 1 = <<5*1=5>>
    Total cost: 6 + 5 = <<6+5=11>>
    #### 11  

<Example 2>
    <Input> Thomas made 4 stacks of wooden blocks. The first stack was 7 blocks tall. The second stack was 3 blocks taller than the first. The third stack was 6 blocks shorter than the second stack, and the fourth stack was 10 blocks taller than the third stack. If the fifth stack has twice as many blocks as the second stack, how many blocks did Thomas use in all?
    <Response>
    The second stack has 7 blocks + 3 blocks = <<7+3=10>>10 blocks.
    The third stack has 10 blocks - 6 blocks = <<10-6=4>>4 blocks.
    The fourth stack has 4 blocks + 10 blocks = <<4+10=14>>14 blocks.
    The fifth stack has 10 blocks x 2 = <<10*2=20>>20 blocks.
    In total there are 7 blocks + 10 blocks + 4 blocks + 14 blocks + 20 blocks = <<7+10+4+14+20=55>>55 blocks.
    #### 55

</SYSTEM>''',

        'captioning': '''<TASK> captioning
<Instruction> Describe the provided image in rich detail. Focus on the main subjects, their attributes (like color and clothing), their actions or interactions, the background setting, and the overall composition of the scene.

<Example 1> 
    [Image: A picture of a golden retriever playing in a park]
    <Response> The image is the cover of a book titled "Devil's Highway" by Hank Janson. The cover features a man in a blue suit and a woman in a pink dress. The man is holding a gun and the woman is looking up at him with a surprised expression on her face. The title of the book is written in red and yellow letters at the top of the cover, with the subtitle "Nine Million Sale" written in smaller letters below. The number "26" is written on the bottom right corner. The background is a dark green color.

<Example 2> 
    [Image: A black and white, comic-style illustration of a car driving forward with a dynamic, radiating background.]
    <Response> The image is a black and white illustration of a car driving on a road. The car is in the center of the image and is facing towards the right side of the frame. It has a large front grille with the Toyota logo on it. There are two people sitting in the driver's seat, one of them is holding a steering wheel and the other is looking out the window. The background is filled with rays of light, creating a sunburst effect. The overall style of the illustration is cartoon-like and playful.

</SYSTEM>''',

        'vqa': '''<TASK> vqa
<Instruction> Answer the question about the image in a single word or a very short phrase.
<Format> single word / very short phrase, not long sentence. Don't attach 'Response' keyword.
<Example 1>
    [Image: A picture of three red apples on a table]
    <Question> How many apples are there?
    <Response> Three

<Example 2>
    [Image: A person sitting on a stool and playing an acoustic guitar]
    <Question> What is the person doing?
    <Response> Playing the guitar

</SYSTEM>''', 

        'text_qa': '''<TASK> text-qa  
<Instruction> Based on the provided input, answer the raw text, separate them with a comma (,)
<Important> DO NOT repeat the questions in your output!! 
<Format> answer1, answer2, answer3 ...

<Example 1>   
    <Input> The patient showed a high fever of 39 degrees Celsius. The doctor prescribed medicine. After taking the medicine, his condition improved.
    <Question> ["What was the patient's temperature?", "Did the medicine work?"]
    <Response> 39 degrees Celsius, yes

<Example 2>
    <Input> (CNN) -- Three American college students detained in Cairo since Monday night were released from police custody Friday and were headed to the airport to return to the United States, an attorney for one of the men said. 
The men will board three separate commercial flights to return home, according to Joy Sweeney, the mother of Derrik Sweeney. 
Theodore Simon, an attorney for the family of Gregory Porter, told CNN that "his parents anxiously await his return." 
The three -- Porter, Sweeney and Luke Gates -- were arrested after being accused of throwing Molotov cocktails in the unrest that has rattled the country since last week. Their release was ordered Thursday. 
Joy Sweeney said earlier Friday that the paperwork to release the men had been completed. Derrik Sweeney's father, Kevin Sweeney, told CNN his flight is scheduled to leave Cairo at 10:30 a.m. Saturday (3:30 a.m. ET) and he will arrive in his home state of Missouri on Saturday night. 
"He's extremely excited," Kevin Sweeney said of his son. The family was planning to hold a belated Thanksgiving meal Sunday. 
Joy Sweeney said her son told her Wednesday in a telephone call that "they had done nothing wrong." All had been attending American University in Cairo on a semester-long, study-abroad program. 
Sweeney, 19, is a Georgetown University student from Jefferson City, Missouri; Porter, 19, is from Glenside, Pennsylvania, and attends Drexel University in Philadelphia; and Gates, 21, of Bloomington, Indiana, goes to Indiana University. 
Adel Saeed, the general prosecutor's spokesman, said Wednesday that a bag filled with empty bottles, a bottle of gasoline, a towel and a camera had been found with the three American students.
    <Question> ['What are the three students names?', 'Where were they detained?', 'Are they all flying in the same flight?', 'What were they accused of?', 'What day was their release ordered?', 'From what state is Derek Sweeney from?', 'What is his family planning to do upon his arrival?', "What are the three men's ages?", "What's the general prosecutor's spokesman's name?", 'Who attends the Indiana university?']
    <Response> Porter, Sweeney and Luke Gates, Cairo, No, throwing Molotov cocktails, Friday, Missouri, They plan to hold a belated Thanksgiving meal., Two of them are 19 years old and the other is 21., Adel Saeed., Luke Gates
</SYSTEM>'''
    }

    task_prompt = task_specific_prompts.get(task)
    return base_system + task_prompt

def prompt_three_shot(task):
    base_system = '''<SYSTEM> You are a helpful multimodal assistant. Analyze the user's query to identify the specific <USER TASK> and generate a response that strictly follows the format shown in the corresponding example below.
<OUTPUT_RULE> IMPORTANT: You MUST NOT include the <Response> tag.
'''
    
    task_specific_prompts = {
        'summarization': '''<TASK> summarization
<Instruction> Provide a high-level summary of the document's primary function and purpose. Focus on what the document establishes, requires, and authorizes, except procedural details.
<Example 1>
    <Input> [A legal text about restructuring the federal budget.]
    <Response> Federal Budget Structure Act of 1993 - Amends Federal law to require that the budget the President submits to the Congress be a unified budget comprising an operating budget and a capital budget, each presented separately for total funds, Federal funds, and trust funds.  Restricts the capital budget to the major activities, projects, and programs supporting the acquisition, construction, alteration, and rehabilitation of capital assets.  Includes all other items in the operating budget. 
Requires the following reports to specified congressional committees on capital activities and operating activities associated with:  (1) roadways and bridges, airports and airway facilities, and mass transportation systems; (2) waste water treatment and related facilities; (3) water resource projects; and (4) public buildings.

<Example 2>
    <Input> [A legal text amending the National Flood Insurance Act to adjust the phase-in period for premium rate increases.] 
    <Response> Saving Homeowners from Onerous Rate Escalations Act of 2013 or SHORE Act of 2013 - Amends the National Flood Insurance Act of 1968 to direct the Administrator of the Federal Emergency Management Agency (FEMA) to phase in, over an eight-year period, any increase in the flood insurance risk premium rate caused by the prohibition against extending subsidies to new or lapsed policies.  Extends from a 5-year to a 10-year period the phase-in period for premium adjustment increases in the flood insurance risk rate. Prescribes a phase-in rate of: (1) 5% for each of the first 5 years after the effective date of an update, and 15% for each of the 5 ensuing years; and (2) 5% for each of the first 5 years following the effective date of designation as a special flood area of any area not previously so designated, and 15% for each of the 5 ensuing years.
    
<Example 3>
    <Input> [A legal text about requiring the Western Area Power Administration to publicly disclose financial and operational information.]
    <Response> Western Area Power Administration Transparency Act (Sec.2)This bill directs the Western Area Power Administration (WAPA)to establish a pilot project to provide increased transparency for its customers. WAPA must publicly display on its website specific information dating back to FY2008, including rates charged by power systems to customers for power and transmission services, the amount of capacity or energy sold by power systems, and a detailed accounting at the functional and budget activity level of all its expenditures and capital costs by region and for the headquarters office. Additionally, WAPA must annually update the information it provides on the website, including the changes it publishes, the reasons for the changes, and the amount of the unobligated balances it retains at the end of the prior fiscal year within each marketing area and at headquarters. The pilot project shall terminate in seven years.
    
</SYSTEM>''',

        'math_reasoning': '''<TASK> math_reasoning
<Instruction> Solve the math problem by your reasoning step-by-step. You must show all reasoning steps. 
<Format> Provide your all reasoning steps and final answer with "####" (4 hash) followed by the numerical result.
<Example 1> 
    <Input> A grocery store sells apples for $2 each and bananas for $1 each. If I buy 3 apples and 5 bananas, how much do I spend in total?
    <Response> 
    I need to calculate the total cost of the apples and the bananas. 
    Cost of apples: 3 * 2 = <<3*2=6>> 
    Cost of bananas: 5 * 1 = <<5*1=5>>
    Total cost: 6 + 5 = <<6+5=11>>
    #### 11  

<Example 2>
    <Input> Thomas made 4 stacks of wooden blocks. The first stack was 7 blocks tall. The second stack was 3 blocks taller than the first. The third stack was 6 blocks shorter than the second stack, and the fourth stack was 10 blocks taller than the third stack. If the fifth stack has twice as many blocks as the second stack, how many blocks did Thomas use in all?
    <Response>
    The second stack has 7 blocks + 3 blocks = <<7+3=10>>10 blocks.
    The third stack has 10 blocks - 6 blocks = <<10-6=4>>4 blocks.
    The fourth stack has 4 blocks + 10 blocks = <<4+10=14>>14 blocks.
    The fifth stack has 10 blocks x 2 = <<10*2=20>>20 blocks.
    In total there are 7 blocks + 10 blocks + 4 blocks + 14 blocks + 20 blocks = <<7+10+4+14+20=55>>55 blocks.
    #### 55

<Example 3>
    <Input> Olaf collects colorful toy cars. At first, his collection consisted of 150 cars. His family, knowing his hobby, decided to give him some toy cars. Grandpa gave Olaf twice as many toy cars as the uncle. Dad gave Olaf 10 toy cars, 5 less than Mum. Auntie gave Olaf 6 toy cars, 1 more than the uncle. How many toy cars does Olaf have in total, after receiving all these gifts?
 
    <Response>
    Dad gave Olaf 10 toy cars,
    Mom has given Olaf 5 more toy cars than Dad, so 10 + 5 = <<10+5=15>>15 toy cars
    Auntie gave Olaf 6 toy cars,
    Uncle has given 1 less toy than Auntie, so 6 - 1 = <<6-1=5>>5 toy cars
    Grandpa gave Olaf 2 * 5 = <<2*5=10>>10 toy cars.
    All the family together gave Olaf 10 +15 + 6 + 5 + 10 = <<10+15+6+5+10=46>>46.
    Adding the cars Olaf already had, Olaf's collection has 150 + 46 = <<150+46=196>>196 cars.
    #### 196
    
    </SYSTEM>''',
    
        'captioning': '''<TASK> captioning
<Instruction> Describe the provided image in rich detail. Focus on the main subjects, their attributes (like color and clothing), their actions or interactions, the background setting, and the overall composition of the scene.

<Example 1> 
    [Image: A picture of a golden retriever playing in a park]
    <Response> The image is the cover of a book titled "Devil's Highway" by Hank Janson. The cover features a man in a blue suit and a woman in a pink dress. The man is holding a gun and the woman is looking up at him with a surprised expression on her face. The title of the book is written in red and yellow letters at the top of the cover, with the subtitle "Nine Million Sale" written in smaller letters below. The number "26" is written on the bottom right corner. The background is a dark green color.

<Example 2> 
    [Image: A black and white, comic-style illustration of a car driving forward with a dynamic, radiating background.]
    <Response> The image is a black and white illustration of a car driving on a road. The car is in the center of the image and is facing towards the right side of the frame. It has a large front grille with the Toyota logo on it. There are two people sitting in the driver's seat, one of them is holding a steering wheel and the other is looking out the window. The background is filled with rays of light, creating a sunburst effect. The overall style of the illustration is cartoon-like and playful.

<Example 3> 
    [Image: A picture of a cover of a book titled "Texas Rangers"]
    <Response> The image is a cover of a book titled ""Texas Rangers"" by Jackson Cole. The cover features a man holding a large gold-colored revolver in his hand, with another man standing behind him, also holding a gun. The background is a dark blue sky with smoke rising from the ground. The title of the book is written in bold white letters at the top of the cover, with the subtitle ""The Fugitive, an action story of two men, hunter and hunted"" written in smaller white letters below. The author's name, ""Tornado Trail"" is written at the bottom.

</SYSTEM>''',

        'vqa': '''<TASK> vqa
<Instruction> Answer the question about the image in a single word or a very short phrase.
<Format> single word / very short phrase, not long sentence. Don't attach 'Response' keyword.
<Example 1>
    [Image: A picture of three red apples on a table]
    <Question> How many apples are there?
    <Response> Three

<Example 2>
    [Image: A person sitting on a stool and playing an acoustic guitar]
    <Question> What is the person doing?
    <Response> Playing the guitar

<Example 3>
    [Image: A library with many books on shelves]
    <Question> How many people are reading?
    <Response> Two
</SYSTEM>''', 

        'text_qa': '''<TASK> text-qa  
<Instruction> Based on the provided input, answer the raw text, separate them with a comma (,)
<Important> DO NOT repeat the questions in your output!! 
<Format> answer1, answer2, answer3 ...

<Example 1>   
    <Input> The patient showed a high fever of 39 degrees Celsius. The doctor prescribed medicine. After taking the medicine, his condition improved.
    <Question> ["What was the patient's temperature?", "Did the medicine work?"]
    <Response> 39 degrees Celsius, yes

<Example 2>
    <Input> (CNN) -- Three American college students detained in Cairo since Monday night were released from police custody Friday and were headed to the airport to return to the United States, an attorney for one of the men said. 
The men will board three separate commercial flights to return home, according to Joy Sweeney, the mother of Derrik Sweeney. 
Theodore Simon, an attorney for the family of Gregory Porter, told CNN that "his parents anxiously await his return." 
The three -- Porter, Sweeney and Luke Gates -- were arrested after being accused of throwing Molotov cocktails in the unrest that has rattled the country since last week. Their release was ordered Thursday. 
Joy Sweeney said earlier Friday that the paperwork to release the men had been completed. Derrik Sweeney's father, Kevin Sweeney, told CNN his flight is scheduled to leave Cairo at 10:30 a.m. Saturday (3:30 a.m. ET) and he will arrive in his home state of Missouri on Saturday night. 
"He's extremely excited," Kevin Sweeney said of his son. The family was planning to hold a belated Thanksgiving meal Sunday. 
Joy Sweeney said her son told her Wednesday in a telephone call that "they had done nothing wrong." All had been attending American University in Cairo on a semester-long, study-abroad program. 
Sweeney, 19, is a Georgetown University student from Jefferson City, Missouri; Porter, 19, is from Glenside, Pennsylvania, and attends Drexel University in Philadelphia; and Gates, 21, of Bloomington, Indiana, goes to Indiana University. 
Adel Saeed, the general prosecutor's spokesman, said Wednesday that a bag filled with empty bottles, a bottle of gasoline, a towel and a camera had been found with the three American students.
    <Question> ['What are the three students names?', 'Where were they detained?', 'Are they all flying in the same flight?', 'What were they accused of?', 'What day was their release ordered?', 'From what state is Derek Sweeney from?', 'What is his family planning to do upon his arrival?', "What are the three men's ages?", "What's the general prosecutor's spokesman's name?", 'Who attends the Indiana university?']
    <Response> Porter, Sweeney and Luke Gates, Cairo, No, throwing Molotov cocktails, Friday, Missouri, They plan to hold a belated Thanksgiving meal., Two of them are 19 years old and the other is 21., Adel Saeed., Luke Gates

<Example 3>
    <Input> 
    CHAPTER XI. 
THE STORM IN THE VALLEY. 
Judging from appearances, when they entered the new cabin of the moonlighters, Ralph concluded that George had said some hard things to Bob because of the part he had obliged him to play. When the two went in to get the few hours of sleep they needed so sadly, for they had been awake during all of the previous night, no one spoke. They were all having what Ralph afterward described as a grand sulking match; but neither one of their guests paid the slightest attention to their ill humor. 
It was then very late in the night, and, tired as each one was, it was but a few moments before the camp was in a state of complete repose, from which neither moonlighter, engineer nor student awakened until the sun had been looking in upon them nearly an hour. 
If Bob had been cross the previous evening, his sleep had restored him to his usual good humor, and he greeted Ralph and George with the cheeriest of smiles. 
""I say, old fellow,"" he began, when Harnett returned from making his toilet at the brook-side, ""I realize that we played you a dirty kind of a trick in using your team as we did last night; but at the time I was so anxious to get everything over here all right that I did not stop to think about it. Of course, I can't undo what has been done, but if any money trouble comes to you because of last night's work, neither you nor Gurney shall lose a cent. Try to forget it, won't you, George? Shake hands with me, and say that you will." 
    <Question> [had they gotten much sleep the previous night?, who was at the Cabin?, who said some hard things to Bob?, Did Bob feel better after he slept?, how did Ralph describe how they were acting?, how did Bob greet Ralph and George?, how long had the sun been up before they awoke?, when did Harnett get back?, where?, did George talk to him?, did he try to apologize?, "who wouldn't lose a cent?", could what George did be undone?, what is the title of chapter 11?]
    <Response> no, Ralph, George and Bob, George, yes, a grand sulking match, with the cheeriest of smiles, nearly an hour, after making his toilet, at the brook-side, yes, yes, George and Gurney, no, THE STORM IN THE VALLEY

</SYSTEM>'''
    }

    task_prompt = task_specific_prompts.get(task)
    return base_system + task_prompt

def get_user_query(task, input_text="", question_text=""):

    user_query = f'''<USER_QUERY>
  <USER_TASK>{task}</USER_TASK>
  <USER_INPUT>'''
    
    if task in ['summarization', 'math_reasoning', 'text_qa']:
        # 텍스트 task
        user_query += f'''
    <Input>{input_text}</Input>'''
        
        if task == 'text_qa' and question_text:
            user_query += f'''
    <Question>{question_text}</Question>'''
    
    elif task == 'vqa':
        user_query += f'''
    <Question>{question_text}</Question>'''
    
    user_query += '''
  </USER_INPUT>
</USER_QUERY>
''' 
    return user_query

def load_successful_urls():
    successful_urls_path = os.path.join(DOWNLOAD_REPORT_PATH, "successful_urls.txt")
    
    if not os.path.exists(successful_urls_path):
        print('다운로드 이미지 먼저 실행하시오')
        return set()
    
    successful_urls = set()
    with open(successful_urls_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                successful_urls.add(url)
    
    print(f"다운로드 성공한 URL : {len(successful_urls)}개")
    return successful_urls

# messages 포맷으로 반환
def format_inference_sample(sample, prompt_function=prompt_three_shot):

    task = sample['task']
    input_text = str(sample.get('input', '')) if sample.get('input') is not None else ''
    question_text = str(sample.get('question', '')) if sample.get('question') is not None else ''
    
    # System Prompt
    system_content = prompt_function(task)
    
    # Message
    message = [
        {"role": "system", "content": [{"type": "text", "text": system_content}]}
    ]

    # User Query
    user_content = [{'type':'text', 'text': get_user_query(task, input_text, question_text)}]

    # Image Case
    if (sample.get('image') is not None and str(sample['image']).strip() != ""):
        user_content.append({"type": "image", "image": sample['image']}) 
    
    message.append({"role": "user", "content": user_content})

    return message

# URL to Cache
def _url_to_cache_path(url: str) -> str:
    parsed = url.split("?")[0]
    ext = os.path.splitext(parsed)[1] if "." in parsed else ".jpg"
    name = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(IMAGE_CACHE_DIR, name + ext)

def create_inference_dataset(successful_urls, output_path=FILE_PATH + 'preprocessed_inference', use_sample=False, prompt_function=prompt_three_shot):
    
    # data path
    data_path = SAMPLE_PATH if use_sample else TEST_PATH
    dataset_type = "샘플" if use_sample else "테스트"
    
    # data load 
    dataset = load_dataset('parquet', data_files = FILE_PATH + data_path, split='train')
    print(f" {dataset_type} 데이터 로드, Sample {len(dataset)}개 ")     
    inference_samples = []
    failed_count = 0
    image_success = 0
    image_failed = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="Preprocessing - Inference")):
        try:
            # Information
            inference_sample = {
                'id': i,
                'task': sample['task'],
                'input_type': sample['input_type'],
                'input': sample['input'],
                'question': sample.get('question', ''),
                'image': None 
            }
            
            # Image Processing
            if sample['input_type'] == 'image':
                input_data = sample['input']

                # input_data가 None이거나 빈 문자열
                if input_data is None or input_data == "":
                    print('이미지 데이터가 비어있음')
                    image_failed += 1

                # 이미지가 URL 형식인 경우 
                elif isinstance(input_data, str) and input_data.startswith("http"):
                    # 캐시된 이미지 경로 확인
                    if input_data in successful_urls:
                        cache_path = _url_to_cache_path(input_data)
                        if os.path.exists(cache_path):
                            inference_sample['image'] = cache_path 
                            image_success += 1
                        else:
                            print('로컬 캐시 이미지 로드에 실패')
                            image_failed += 1

                    else:
                        print('아직 다운받지 않은 이미지 경로임. http 리퀘스트로 시도해야할 수 있다.')
                        image_failed += 1

                # Base64 format - 그대로 넣어서 나중에 처리 
                else:
                    inference_sample['image'] = input_data
                    image_success += 1

            # Actual Sample Information
            actual_sample = {
                'id': i,
                'messages': None
            }
                    
            # 추론용 메시지 생성 - prompt_function 전달
            actual_sample['messages'] = format_inference_sample(inference_sample, prompt_function)
            inference_samples.append(actual_sample)

        except Exception as e:
            print(f" 샘플 {i} 처리 실패 - {e}")
            failed_count += 1
            continue
    
    print(f"[ 전처리 완료 ]")
    print(f" 성공 : {len(inference_samples)}, 실패 : {failed_count}")
    print(f" 이미지 - 성공 : {image_success}, 실패 : {image_failed}")

    inference_dataset = Dataset.from_list(inference_samples)
    os.makedirs(output_path, exist_ok=True)

    # Save
    dataset_filename = "sample_dataset" if use_sample else "inference_dataset"
    dataset_save_path = os.path.join(output_path, dataset_filename)
    inference_dataset.save_to_disk(dataset_save_path)
    
    # 메타데이터 
    metadata = {
        'total_samples': len(inference_samples),
        'failed_samples': failed_count,
        'image_success': image_success,
        'image_failed': image_failed,
        'preprocessing_type': 'inference_optimized',
        'data_source': 'sample' if use_sample else 'test',
        'data_path': data_path,
        'prompt_mode': prompt_function.__name__ 
    }
    
    metadata_filename = "sample_metadata.json" if use_sample else "inference_metadata.json"
    metadata_path = os.path.join(output_path, metadata_filename)
    
    # DEBUG
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return inference_dataset

def main():
    parser = argparse.ArgumentParser(description='Preprocessing - Inference')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--mode', choices=['one', 'two', 'three'], default='three')
    args = parser.parse_args()
    
    # 프롬프트 함수 매핑
    prompt_functions = {
        'one': prompt_one_shot,
        'two': prompt_two_shot,
        'three': prompt_three_shot
    }
    
    selected_prompt_function = prompt_functions[args.mode]

    # 성공 URL 로드
    successful_urls = load_successful_urls()
    if not successful_urls:
        print('다운로드 이미지가 없음, 먼저 다운로드를 진행하시오')
        exit(0)

    # 출력 경로 설정 (샘플 모드일 때 구분)
    output_path = FILE_PATH + 'preprocessed_inference'

    create_inference_dataset(
        successful_urls=successful_urls, 
        output_path=output_path,
        use_sample=args.sample,
        prompt_function=selected_prompt_function
    )

if __name__ == "__main__":
    main()

