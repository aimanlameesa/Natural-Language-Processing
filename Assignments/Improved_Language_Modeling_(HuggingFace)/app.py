from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

import torch
import transformers
from transformers import pipeline

from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# adding the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

@app.route('/')
def index():
    return redirect(url_for('autocomplete'))


class MyForm(FlaskForm):
    name = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/home', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    code = False
    name = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        pipe = pipeline("text-generation", max_length=100, pad_token_id=0, eos_token_id=0, model="aiman-lameesa/codeparrot-ds-accelerate", tokenizer=tokenizer)
        code = pipe(name, num_return_sequences=50)[0]["generated_text"]
        form.name.data = ""
    return render_template("homepage.html",form=form,name =name, code=code)

if __name__ == "__main__":
    app.run(debug=True)