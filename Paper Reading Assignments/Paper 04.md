## PPT: Pre-trained Prompt Tuning for Few-shot Learning

Paper link: https://arxiv.org/pdf/2109.04332.pdf

| Aim | Gu et al. proposed pre-trained prompts by adding soft prompts during the pre-training stage in order to achieve a better initialization for downstream tasks. | 
| ------- | --- | 
| Background | Prompts for pre-trained language models (PLMs) had been proved very effective since they worked as the connection between pre-training tasks and various downstream tasks. However, prompt tuning freezes PLMs and tunes soft prompts only, thus it is able to provide an efficient solution for adapting large-scale PLMs to downstream tasks. Though prompt tuning performs very well with sufficient downstream data, it is much worse under few-shot learning settings which may hinder the performance of prompt tuning in different applications. In this paper, Gu et al. proposed pre-trained prompt tuning (PPT) to address this limitation. | 
| Datasets | They mostly experimented with English and Chinese datasets, where they presented the results of the T5 model (from small size to XXL) for full-model tuning (FT) and the results of PPT and other baselines under prompt tuning (PT). To be more spefic, they used SST-2, RACE-m, BoolQ, CB datasets for English tasks and CCPM, C<sup>3</sup>, LCQMC etc datasets for Chinese tasks. | 
| Methods | For the work, they pre-trained prompts and used the pre-trained prompts for specific tasks. Given an input sentence and its label, a pattern mapping is first applied to convert the input into a new sequence, whereas the new sequence adds some prompt tokens as hints and preserves the masked tokens at the same time so that the PLMs can predict tokens at the masked positions. Then a verbalizer is used to map the true label to some label tokens. The classification task can be represented by a pattern-verbalizer pair of new sequences and the label tokens.
arg max
θ
X
x
log p

|  
| Results and Findings|  | 
| Limitations | There is still an existing gap between masked language modeling and downstream tasks though prompt pre-training was capable to bridge this gap to some extent. Moreover, pre-trained prompt tuning converges slower than full-model tuning (FT). Therefore, how to further accelerate the convergence needs to explored in future according to this research. |  
| Future Work | The significant future research directions based on this paper are: (1) Designing unified task formats with their corresponding pre-training objectives for different tasks such as, language generation or relation extraction (2) Evaluating the performance of few-shot learning techniques considering other parameter tuning approaches and adapting unified task pre-training to them (3) Studying the performance unified task pre-training on pre-trained language models apart the soft-prompt based approaches | 


