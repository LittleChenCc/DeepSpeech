# Aishell

## Conformer
paddle version: 2.2.2  
paddlespeech version: 0.1.2
| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | attention | - | 0.0548 |
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | ctc_greedy_search | - | 0.05127 |
| conformer | 47.07M  | conf/conformer.yaml | spec_aug| test | ctc_prefix_beam_search | - | 0.05131 | 
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | attention_rescoring | - | 0.04829 | 


## Chunk Conformer
paddle version: 2.2.2  
paddlespeech version: 0.1.2  
Need set `decoding.decoding_chunk_size=16` when decoding.

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | attention | 16, -1 | - | 0.0573884 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | ctc_greedy_search | 16, -1 | - | 0.06599091 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | ctc_prefix_beam_search | 16, -1 | - | 0.065991 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | attention_rescoring | 16, -1 |  - | 0.056502 |  


## Transformer 

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention | 3.8103787302970886 | 0.056588 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_greedy_search | 3.8103787302970886 | 0.059932 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_prefix_beam_search | 3.8103787302970886 | 0.059989 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention_rescoring | 3.8103787302970886 | 0.052273 |  
