## Sentiment Analyzer

We used Flask to create a web-based sentiment analyzer that can scrap reddit data related to a product when we enter the product name. The website also shows the count of positive and negative sentiments for that product and also the top 5 positive and negative words inside those scrapped posts. For this application, one has to simply input a product name to check the sentiments of a product.

For this task, we defined the instances of Reddit using praw. Then we created the necassary frontend using html and css to display the process in a web application. We used "requirements.txt" file which contains all the necessary libraries, created "app.py" to apply Flask and pass the necessary functions in the application.

A few screenshots from the web application are attached below:

### Homepage of the Sentiment Analyzer
![Home Page](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Sentiment%20Analysis/images/homepage.png)

### Entering a Product Name
![Entering Product Name](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Sentiment%20Analysis/images/input.png)

### Obtaining Reddit Posts 
![Obtaining Reddit Posts](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Sentiment%20Analysis/images/posts.png)

### Analysis of Positive and Negative Posts
![Analysis of Positive and Negative Posts](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Sentiment%20Analysis/images/counting.png)

### Top 5 Positive and Negative Words
![Top 5 Positive and Negative Words](https://github.com/aimanlameesa/Natural-Language-Processing/blob/main/Assignments/Sentiment%20Analysis/images/topwords.png)
