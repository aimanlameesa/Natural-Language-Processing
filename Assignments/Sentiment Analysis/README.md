# review_product_web

## How to

1. To scrap twitter post by keyword, I used twint because i can't get Twitter API --> install twint https://github.com/twintproject/twint/wiki/Setup 
    * if you can't install maybe upgrade pip or install visual C++ can fix it https://visualstudio.microsoft.com/visual-cpp-build-tools/
    * 
     ![image](https://user-images.githubusercontent.com/78832408/219311056-8c6537f6-9b7c-461f-b4f4-e360313ff5e4.png)

2. Train model, I used bi-LSTM train with SST2 dataset 



---
## Review

1. input keyword

![image](https://user-images.githubusercontent.com/78832408/219096665-1b0706af-45c4-467d-ac3c-3d6514200733.png)

2. show comparison and top word 

   * I'm not train much and because this is binary classification, model may bias on negative, next time we may try over-sampling and hyperparameter tuning for improve model  
  
![image](https://user-images.githubusercontent.com/78832408/219310666-18d6977d-aff3-485a-8fa5-2e350d84fab2.png)

*bitcoin

![image](https://user-images.githubusercontent.com/78832408/219315763-2a4900c2-3068-4b10-8719-ae343cf34e05.png)

