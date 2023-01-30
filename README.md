# NLP Project

## Graph Enhanced Contrastive Learning for Radiology Findings Summarization

| Aim | Hu et al.proposed an integrated framework to exploit extra knowledge and the original findings simultaneously to address the limitation to facilitate impression generation by extracting critical information appropriately.| 
| ------- | --- | 
| Background | A radiology report contains an impression section which is a summary of the observations from the finding section. It is very important for the radiologists to convey the impression section properly to the physicians. Since the process of summarizing is time-consuming and prone to error for inexperienced radiologists at the same time, automatic generation of impression has attracted substantial attention lately. Though recent studies had used a separate encoder to incorporate extra knowledge, it had not been effective enough. Hu et al.proposed an integrated framework to exploit extra knowledge and the original findings simultaneously to address the limitation. | 
| ---------- | --- | 
| Datasets | They experimented on two datasets: OPENI (Demner-Fushman et al., 2016) and MIMIC-CXR (Johnson et al., 2019), whereas the former dataset contains 3268 reports and the latter one is a larger dataset with 124577 reports. | 
| ------- | --- | 
| Methods | They constructed a word graph for each input findings through the automatically extracted entities and dependency tree, with its embeddings from a text encoder. Then they modeled the relation information among key words through a graph encoder (e.g., graph neural networks (GNNs)). Finally, they introduced contrastive learning to map positive samples (constructed by masking non-key words) closer and push apart negative ones (constructed by masking key words) to emphasize the key words in the findings. | 
| ------- | --- | 
| Results| 301 | 
| ------- | --- |
| Limitations | 301 | 
| ------- | --- | 
| Future Work | 301 | 


A radiology report contains an impression section which is a summary of the observations from the finding section. It is very important for the radiologists to convey the impression section properly to the physicians. Since the process of summarizing is time-consuming and prone to error for inexperienced radiologists at the same time, automatic generation of impression has attracted substantial attention lately. Though recent studies had used a separate encoder to incorporate extra knowledge, it had not been effective enough. Hu et al.proposed an integrated framework to exploit extra knowledge and the original findings simultaneously to address the limitation. To do this, they used a text encoder to encode each input and constructed a graph using its entities and dependency tree. Then, they deployed a graph encoder to model relation information in the constructed graph. Lastly, they introduced contrastive learning to map positive samples closer and push apart negative ones in order to emphasize the key words in the findings.



In detail, for each input findings, it is encoded
by a text encoder, and a graph is constructed
through its entities and dependency tree. Then,
a graph encoder (e.g., graph neural networks
(GNNs)) is adopted to model relation information in the constructed graph. Finally, to emphasize the key words in the findings, contrastive
learning is introduced to map positive samples
(constructed by masking non-key words) closer
and push apart negative ones (constructed by
masking key words). The experimental results
on OpenI and MIMIC-CXR confirm the effectiveness of our proposed method
