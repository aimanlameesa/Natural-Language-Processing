import pandas as pd
import numpy as np

import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

from PyPDF2 import PdfReader

nlp = spacy.load('en_core_web_md')
skill_path = "static/pattern_data/education_skill.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)

# doc = nlp("Chaky loves deep learning.")
# doc.ents

# writing a function to preprocess the data
def preprocessing(sentence):
    
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM':
                cleaned_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(cleaned_tokens)

# writing a function to obtain the skills and education
def skills_and_education(text):
    # passing the text to the nlp
    doc = nlp(text)  # note that this nlp already know skills
    
    skills = []
    education = []
    
    # looking at the ents
    for ent in doc.ents:
        # if the ent.label_ is SKILL, then we append to some list
        if ent.label_ == "SKILL":
            skills.append(ent.text)
        if ent.label_ == "EDUCATION":
            education.append(ent.text)

    # converting to list 
    skills_list = list(set(skills))
    education_list = list(set(education))

    # reversing education list to display in order
    education_list.reverse()

    return skills_list, education_list


# writing a function to extract skills and education from the resume
def extract(filePath):
    reader = PdfReader(filePath)
    number_of_pages = len(reader.pages)

    # extracting text from all pages of a resume 
    text = ''
    for page_number in range(number_of_pages):
        page = reader.pages[page_number]
        text += ' ' + page.extract_text() 
    
    # preprocessing text
    text = preprocessing(text)
    doc = nlp(text)

    # extracting unique skills and eduation
    skills, education = skills_and_education(doc)

    return skills, education