## Bengali-English Translation

We used Flask to create a web-based Bengali to English Translator given a source sentence in Bengali. For this application, one has to input a Bengali sentence and the program will automatically predict the English translation of that sentence.

For this Bengali to English Translation task, we used the "ben-eng" dataset (from http://www.manythings.org/anki/) and used encoder-decoder with "attention" to predict the English translation of a given Bengali sentence. Then we created the necassary frontend using html and css to display the process in a web application. We used "requirements.txt" file which contains all the necessary libraries, created "main.py" to apply Flask and pass the necessary functions in the application.

A few screenshots from the web application are attached below:

### Homepage of the Bengali to English Translator
![Home Page](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Bengali-English%20Translation/images/homepage.png)

### Entering a Bengali Sentence
![Entering Bengali Sentence](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Bengali-English%20Translation/images/input.png)

### Predictive Sentence in English
![Predictive Sentence English](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Bengali-English%20Translation/images/output.png)
