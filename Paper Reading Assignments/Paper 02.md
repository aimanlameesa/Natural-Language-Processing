## Discriminative Marginalized Probabilistic Neural Method for Multi-Document Summarization of Medical Literature

Paper link: https://aclanthology.org/2022.acl-long.15.pdf

| Aim |  Moro et al. proposed a novel framework to discriminate critical information from a cluster of topic-related medical documents and generate a multi-document summary. | 
| ------- | --- | 
| Background | The previous studies focused on mainly two approaches : (1) hierarchical networks capturing cross-document relations via graph encodings and (2) long-range neural models carrying out multi-input concatenation. But these approaches struggled to handle clusters of topic-related documents with low computational resources. Moreover, the pre-trained Transformers used to perform well in downstream tasks such as single-document summarization. Since processing a multi-document summary requires to have high capabilities to discriminate the correct information from the clusters merge them consistently, Moro et al. introduced a discriminative marginalized probabilistic neural method (DAMEN) to address this problem. | 
| Datasets | The proposed method was tested and evaluated on the the MS2 dataset to generate systematic literature reviews. The dataset contains more than 470K document abstracts and 20K summaries derived from the scientific literature. Each sample from the dataset consists of three common elements, which are (1) the background statement (to be more specific, a short text describing the research question), (2) the target statement (basically the multi-document summary to generate) and (3) the studies (which is a set of abstracts of topic-related medical studies covered in the background statement). | 
| Methods |  |  
| Results and Findings|  | 
| Limitations | |  
| Future Work | Further research can be conducted to extract relevant snippets from documents with different weighting techniques, semantic relations with unsupervised methods or apply event extraction methods with more explainability and interpretability. Moreover, models can be trained to write and read cross-documents using self-supervised learning methods and memory-based neural networks to deal with multiple inputs. | 
