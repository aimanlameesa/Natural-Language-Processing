# NLP Paper Readings

## Graph Enhanced Contrastive Learning for Radiology Findings Summarization

Paper link: https://arxiv.org/pdf/2204.00203.pdf

| Aim | Hu et al.proposed an integrated framework to exploit extra knowledge and the original findings simultaneously to facilitate impression generation by extracting critical information appropriately.| 
| ------- | --- | 
| Background | A radiology report contains an impression section which is a summary of the most prominent observations from the finding section. It is very important for the radiologists to convey the impression section properly to the physicians. Since the process of summarizing is time-consuming and prone to error for inexperienced radiologists at the same time, automatic generation of impression had attracted substantial attention for research. Though existing studies had used a separate encoder to incorporate extra knowledge, it had not been effective enough. That is why, Hu et al. proposed an integrated framework to address this limitation in this paper. | 
| Datasets | They experimented on two datasets: OPENI (Demner-Fushman et al., 2016) and MIMIC-CXR (Johnson et al., 2019), whereas the first dataset contains 3268 reports and the second one is a larger dataset with 124577 reports. | 
| Methods | They constructed a word graph for each input findings through the automatically extracted entities and dependency tree, and used the embeddings from a text encoder. Then they modeled the relation information among key words using a graph encoder. Finally, they introduced contrastive learning to map positive samples closer and push apart negative ones to emphasize the key words in the findings. |  
| Results and Findings| Effect of Graph and Contrastive Learning (CL) <br> 1. Both BASE+GRAPH and BASE+CL could achieve better results than BASE, indicating that graph and contrastive learning can improve impression generation respectively <br> 2. BASE+GRAPH+CL outperformed all baselines with significant improvements on two datasets, confirming the effectiveness of the proposed method <br> 3. When comparing the two datasets, the performances of the proposed model on OpenI were more prominent than the MIMIC-CXR, maybe due to the smaller size and a shorter averaged word-based length of OpenI dataset <br> 4. For the FC metric on the MIMIC-CXR dataset, a higher F1 score indicated that the complete model could generate more accurate impressions <br> Comparison with Previous Studies <br> 1. The model achieved better performance than ONTOLOGYABS model <br> 2. It outperformed all existing models in terms of F1 scores <br> 3. It could achieve better performance through more straightforward method compared to complicated models <br> Human Evaluation <br> 1. Compared to to BASE, the model outperformed it on four metrics, where 16% key, 25% readability, 18% accuracy and 8% completeness of impressions from the model obtained higher quality than BASE <br> 2. Compared against reference impressions, the model was able to obtain 86%, 78%, and 92% similar results on key, accuracy and completeness as the radiologists, while it was less preferred for readability with a 10% gap <br> Findings Length <br> 1. The performance of BASE and the proposed model decreased as the findings length became long <br> 2. The model outperformed BASE in all the groups, regardless of the findings length <br> 3. A grey line with a downward trend was observed indicating that the model might gain better improvements over BASE on shorter findings than that on longer ones | 
| Limitations | While comparing the proposed model against reference impressions, it could obtain good results on key, accuracy, and completeness but it is not preferred for readability since the metric had around 10% gap. The main reason could be, many removed words in positive examples were used to keep sequence fluently, and the model showed a tendency to identify them as secondary information.|  
| Future Work | N/A | 

## Discriminative Marginalized Probabilistic Neural Method for Multi-Document Summarization of Medical Literature

Paper link: https://aclanthology.org/2022.acl-long.15.pdf

| Aim | | 
| ------- | --- | 
| Background | | 
| Datasets |  | 
| Methods |  |  
| Results and Findings|  | 
| Limitations | |  
| Future Work | | 
