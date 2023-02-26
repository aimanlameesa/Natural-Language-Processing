## Code Autocompletion

We used Flask to create a web-based automatic code generator given a few words of code. For this application, one has to input a few words of code from the beginning and then the program will automatically predict the latter words of code given a fixed sequence length.

For this Language Modeling task, we used a LSTM model to predict the next words in a sequence. Then we created the necassary frontend using html to display the process in a web application. We used "requirements.txt" file which contains all the necessary libraries, created "app.py" to apply Flask and pass the necessary function in the application.

A few screenshots from the web application are attached below:

### Homepage of the Code Generator
![Home Page](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Code%20Autocompletion/images/homepage.png)

### Entering a Few Words of Codes
![Entering Few Words Codes](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Code%20Autocompletion/images/input.png)

### Prediction Based on Given Input Codes
![Prediction Based Given Input Codes](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Code%20Autocompletion/images/output.png)
