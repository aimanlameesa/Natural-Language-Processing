# NLP Project

## Graph Enhanced Contrastive Learning for Radiology Findings Summarization

| Aim | Hu et al.proposed an integrated framework to exploit extra knowledge and the original findings simultaneously to address the limitation to facilitate impression generation by extracting critical information appropriately.| 
| ------- | --- | 
| Background | A radiology report contains an impression section which is a summary of the most prominent observations from the finding section. It is very important for the radiologists to convey the impression section properly to the physicians. Since the process of summarizing is time-consuming and prone to error for inexperienced radiologists at the same time, automatic generation of impression had attracted substantial attention for research. Though existing studies had used a separate encoder to incorporate extra knowledge, it had not been effective enough. That is why, Hu et al.proposed an integrated framework to address the limitation in this paper. | 
| Datasets | They experimented on two datasets: OPENI (Demner-Fushman et al., 2016) and MIMIC-CXR (Johnson et al., 2019), whereas the former dataset contains 3268 reports and the latter one is a larger dataset with 124577 reports. | 
| Methods | They constructed a word graph for each input findings through the automatically extracted entities and dependency tree, with its embeddings from a text encoder. Then they modeled the relation information among key words through a graph encoder (e.g., graph neural networks (GNNs)). Finally, they introduced contrastive learning to map positive samples (constructed by masking non-key words) closer and push apart negative ones (constructed by masking key words) to emphasize the key words in the findings. |  
| Results and Findings| # Effect of Graph and Contrastive Learning (CL)
1. Both BASE+GRAPH and BASE+CL could achieve better results than BASE, which indicates that graph and contrastive learning can respectively promote impression generation
2. BASE+GRAPH+CL outperformed all baselines with significant improvement on two datasets, confirming the effectiveness of the proposed method 
3. When comparing these two datasets, the performance gains of the proposed model on OpenI re more prominent than the MIMIC-CXR, maybe due to small size and a shorter averaged word-based length of OpenI dataset
# Comparison with Previous Studies | 
| Limitations | While comparing the proposed model against reference impressions, it could obtain good results on key, accuracy, and completeness, with 86%, 78%, and 92% similar to radiologists, while it is less preferred for readability with a 10% gap. The main reason could be, many removed words in positive examples were used to keep sequence fluently, and the model showed a tendency to identify them as secondary information.|  
| Future Work | 301 | 

