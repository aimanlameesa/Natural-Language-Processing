# -*- coding: utf-8 -*-
"""resume_parser.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U05An5ro0wAiymVfSy8PfGoSGOIWxe9t
"""

# !pip install spacy or pip install -U 'spacy[cuda-autodetect]'

# !python -m spacy download en_core_web_sm #trained using cnn

# !python -m spacy download en_core_web_md #has word embedding (gloVe); trained using cnn

# !python -m spacy download en_core_web_trf #everything is trained using transformer

"""## 1. Loading Data"""

import pandas as pd
import numpy as np

df_resume = pd.read_csv("/content/resume.csv")

df_resume = df_resume.reindex(np.random.permutation(df_resume.index))
# df_resume = df_resume.copy().iloc[:1000, ]  # optional if your computer is fast, no need
df_resume = df_resume.copy()
df_resume.shape

"""## 2. Loading Skills and Education Data"""

import spacy

nlp = spacy.load('en_core_web_md')
skill_path = "/content/education_skill.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)
nlp.pipe_names

doc = nlp("Chaky loves deep learning.")
doc.ents

"""## 3. Extracting Skills and Education """

df_resume.head()

from spacy.lang.en.stop_words import STOP_WORDS

# before that, let's clean our resume.csv dataframe
def preprocessing(sentence):
    
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM':
                cleaned_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(cleaned_tokens)

# random sampling
random_resume = df_resume.Resume_str.iloc[5]
random_resume[:300]

preprocessing(random_resume[:300])

# let's apply to the whole dataframe
for i, row in df_resume.iterrows():
    clean_text = preprocessing(row.Resume_str)
    df_resume.at[i, 'Clean_resume'] = clean_text

df_resume.head()

"""## 4. Let's really Extract Skills and Education!!"""

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

# def unique_skills(x):
    # return list(set(x))

df_resume.head(1)

df_resume['Skills_Education'] = df_resume.Clean_resume.apply(skills_and_education)
# df_resume['Skills'] = df_resume.Skills.apply(unique_skills)

df_resume.Skills_Education.iloc[0]

"""## 5. Visualization"""

set(df_resume.Category)

category = 'INFORMATION-TECHNOLOGY'
cond = df_resume.Category == category

df_resume_it = df_resume[cond]
df_resume_it.shape

# skills_education = np.concatenate(df_resume_it.Skills_Education.values)

# counting
# from collections import Counter, OrderedDict

# counting = Counter(skills_education)
# counting = OrderedDict(counting.most_common(10))

# counting

# counting.shape

# import matplotlib.pyplot as plt

# plt.figure(figsize=(15, 3))
# plt.xticks(rotation =45)

# plt.bar(counting.keys(), counting.values())

"""## 6. Name Entity Recognition"""

from spacy import displacy

text = df_resume_it.Clean_resume.iloc[43]

doc = nlp(text)

nlp.pipe_names

colors = {"SKILL": "linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c)", 
          "EDUCATION": "linear-gradient(#e66465, #9198e5);"}
options = {"colors": colors}

displacy.render(doc, style="ent", options=options, jupyter=True)

"""## 7. Let's Load the PDF (adding some realism)"""

# ! pip install PyPDF2

from PyPDF2 import PdfReader
# pip install PyPDF2

reader = PdfReader("/content/someone_cv.pdf")
page = reader.pages[0] #first page just for demo
text = page.extract_text()

text = preprocessing(text)

doc = nlp(text)

colors = {"SKILL": "linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c)", 
          "EDUCATION": "linear-gradient(#e66465, #9198e5);"}
options = {"colors": colors}

displacy.render(doc, style="ent", options=options, jupyter=True)

# collecting all the skills and education and put it into a list

skills = []
education = []

for ent in doc.ents:
    if ent.label_ == 'SKILL':
        skills.append(ent.text)
    if ent.label_ == 'EDUCATION':
        education.append(ent.text)
        
# print(set(skills))
# print(set(education))