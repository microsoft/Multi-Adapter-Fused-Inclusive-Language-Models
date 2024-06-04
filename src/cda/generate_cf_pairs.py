import os
import openai
from tqdm import tqdm
from transformers import AutoTokenizer
import pdb
import time
from ast import literal_eval
from collections import defaultdict as dd


#new key
openai.api_key = "xxxxxx"
openai.api_base = "https://gcrgpt4aoai6.openai.azure.com/"#"https://gpttesting1.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = "2023-03-15-preview"#'2022-12-01' # this may change in the future
deployment_id='gpt-35-turbo'#'DaVinci003' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

def get_gender_example_prompt():
    gender_pairs = [
        "(actor, actress)", "(actors, actresses)", "(airman, airwoman)", "(airmen, airwomen)", "(uncle, aunt)", "(uncles, aunts)",
    "(boy, girl)", "(boys, girls)", "(groom, bride)", "(grooms, brides)", "(brother, sister)", "(brothers, sisters)", "(businessman, businesswoman)", "(businessmen, businesswomen)", "(chairman, chairwoman)", "(chairmen, chairwomen)", "(dude, chick)", "(dudes, chicks)", "(dad, mom)", "(dads, moms)", "(daddy, mommy)", "(daddies, mommies)", "(son, daughter)", "(sons, daughters)", "(father, mother)", "(fathers, mothers)", "(male, female)", "(males, females)", "(guy, gal)", "(guys, gals)",
    "(gentleman, lady)", "(gentlemen, ladies)", "(grandson, granddaughter)", "(grandsons, granddaughters)",
    "(guy, girl)", "(guys, girls)", "(he, she)", "(himself, herself)",
    "(him, her)", "(his, her)", "(husband, wife)", "(husbands, wives)", "(king, queen)", "(kings, queens)", "(lord, lady)",
    "(lords, ladies)", "(sir, maam)", "(man, woman)", "(men, women)", "(sir, miss)", "(mr., mrs.)", "(mr., ms.)", "(policeman, policewoman)", "(prince, princess)", "(princes, princesses)", "(spokesman, spokeswoman)", "(spokesmen, spokeswomen)"
    ]
    example_prompt = create_prompt(gender_pairs)
    return example_prompt

def get_race_example_prompt():
    race_triple = ["(black, caucasian, asian)", "(african, caucasian, asian)", "(black, white, asian)"]
    race_pairs = []
    for race in race_triple:
        race = race.strip("(").strip(")").split(", ")
        j=0
        for ele in race:
            j+=1
            for i in range(j,3):
                race_pairs.append((ele,race[i]))
    race_pairs = list(set(race_pairs))
    example_prompt = create_prompt(race_pairs)
    return example_prompt

def get_religion_example_prompt():
    religion_triple = ["(jewish, christian, muslim)",'(jews, christians, muslims)','(torah, bible, quran)','(synagogue, church, mosque)','(rabbi, priest, imam)','(judaism, christianity, islam)']
    religion_pairs = []
    for religion in religion_triple:
        religion = religion.strip("(").strip(")").split(", ")
        j=0
        for ele in religion:
            j+=1
            for i in range(j,3):
                religion_pairs.append((ele,religion[i]))
    religion_pairs = list(set(religion_pairs))
    example_prompt = create_prompt(religion_pairs)
    return example_prompt

def create_prompt(example_pairs):
    example_prompt = "\n ".join([
        f"{pair[0]} -> {pair[1]}"
        for pair in map(lambda x: x.strip("(").strip(")").split(", ") ,example_pairs[:50])
    ])
    return example_prompt

blocked_list = set(["fetishism","Russophilia","sex magic"])
def get_data(bias_type, global_dict, last_fail):
    path = "/home/t-jainprachi/project/data/readable_bias_entity_en_1/"
    if bias_type == "race":
        example_prompt = get_race_example_prompt()
        files = ["readable_race_citizenship_P27.txt", "readable_race_ethnic_group_P172.txt","readable_race_race_Q3254959.txt","readable_race_social_group_Q874405.txt"]
    elif bias_type =="gender":
        example_prompt = get_gender_example_prompt()
        files = ["readable_gender_gender_P21.txt","readable_gender_grammatical-gender_P5185.txt","readable_gender_male_form_of_label_P3321.txt","readable_gender_personal_pronoun_P6553.txt"]
    elif bias_type == "profession":
        example_prompt = get_gender_example_prompt()#get_profession_example_prompt()###ERRR
        files=["readable_profession_field_of_work_P101.txt","readable_profession_occupation_P106.txt","readable_profession_practiced_by_P3095.txt"]
    elif bias_type == "religion":
        example_prompt = get_religion_example_prompt()
        files = ["readable_religion_deity_Q178885.txt","readable_religion_deity_worshipped_P1049.txt","readable_religion_religion_P140.txt","readable_religion_religion_Q9174.txt","readable_religion_religious_festival_Q375011.txt","readable_religion_religious_identity_Q4392985.txt","readable_religion_religious_object_Q21029893.txt","readable_religion_religious_site_Q105889895.txt","readable_religion_religious_text_Q179461.txt","readable_religion_structure_of_worship_Q1370598.txt","readable_religion_world_view_Q71966963.txt","readable_religion_worshipped_by_P1049.txt"]
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    #global_dict = dd(str)
    # if bias_type == "race":
    #     f=open(path+"cf/cf_readable_race_citizenship_P27.txt");L=f.readlines();f.close()
    #     for ele in L:
    #         ele = ele.strip("\n").strip("(").strip(")").split(",")
    #         global_dict[ele[0].strip()] = ele[1].strip()


    for file_name in files:
        missed = [];entities = [];generated_pairs = [];
        with open(path+file_name, "r") as f:
            for line in f:
                line = line.strip("\n")
                if len(line.split("\t")) < 2:
                    continue
                word = line.split("\t")[1]
                try:
                    word_dict = literal_eval(word)
                    if word_dict['language'] == "en":
                        word = word_dict['text']
                    else:
                        word = ""
                except as Exception:
                    word 
                ##

                if word not in entities:
                    if len(word) >= 2:
                        entities.append(word) 
                        token_id = tokenizer.encode(word, add_special_tokens = False)
                        if len(token_id) == 1:
                            token_id = token_id[0]

        # Send a completion call to generate an answer
        print('Sending a test completion job')
        start_phrase = "Here are some examples of counterfactual pairs for different "+bias_type+" terms:\n {} \n {} -> ".format(example_prompt, "TEST QUERY")
        print('Prompt template::', start_phrase)            
        
        i=0
        for gendered_entity in tqdm(entities):
            if gendered_entity in blocked_list:
                missed.append(gendered_entity)
                continue
            if not (gendered_entity in global_dict):
                try:
                    #start_phrase = "Here are some examples of counterfactual pairs for different gendered terms:\n{} \n{} -> ".format(example_prompt, gendered_entity)
                    start_phrase = "Here are some examples of counterfactual pairs for different "+bias_type+" terms:\n {} \n {} -> ".format(example_prompt, gendered_entity)
                    last_fail["word"] = gendered_entity
                    response = openai.Completion.create(engine=deployment_id,
                                                        prompt=start_phrase,
                                                        max_tokens=5,
                                                        temperature = 1,
                                                        top_p = 0.5)
                    answer = response['choices'][0]['text'].strip()
                    answer = answer.split('\n')[0]
                    generated_pairs.append(f"({gendered_entity}, {answer})")
                    global_dict[gendered_entity] = answer

                    time.sleep(0.5)
                    i+=1
                    if i>15:
                        time.sleep(5)#time.sleep(15)
                        i=0
                    
                # except:
                #     print("Err", start_phrase, ":::ent:::", gendered_entity)
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)  
                    print(message)

                    time.sleep(20)
                    # if type(ex).__name__ == "RateLimitError":
                    #    time.sleep(15)
                    
                    start_phrase = "Here are some examples of counterfactual pairs for different "+bias_type+" terms:\n {} \n {} -> ".format(example_prompt, gendered_entity)
                    response = openai.Completion.create(engine=deployment_id,
                                                        prompt=start_phrase,
                                                        max_tokens=5,
                                                        temperature = 1,
                                                        top_p = 0.5)
                    answer = response['choices'][0]['text'].strip()
                    answer = answer.split('\n')[0]
                    print(gendered_entity, "<->", answer,"!")
                    generated_pairs.append(f"({gendered_entity}, {answer})")
                    global_dict[gendered_entity] = answer
                    ##
                except:
                    missed.append(gendered_entity)
            else:
                answer = global_dict[gendered_entity]
                generated_pairs.append(f"({gendered_entity}, {answer})")

        with open(path+"cf/"+"cf_"+file_name, "w") as f:
            f.write("\n".join(generated_pairs))

        with open(path+"cf/"+"missed_cf_"+file_name, "w") as f:
            f.write("\n".join(missed))
    return global_dict

global_dict = dd(str)
last_fail = dd(str)

bias_type = ["race","religion","gender","profession"]
for ele in bias_type:
    global_dict = get_data(ele,global_dict,last_fail)



