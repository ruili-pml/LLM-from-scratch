# LLM-from-scratch

## Table of Contents

### <1> tokenisation
#### notes
`tokenisation.md`
- Unicode, UTF-8
- Byte-pair Encoding 
- Problems caused by tokenisation

#### implementation
`tokeniser.ipynb`
- Unicode, UTF-8
- BPE implementation

### <2> vanilla transformer (GPT-2)
#### notes
`vanilla-transformer.md`
- self-attention
- multi-head attention
- common transformer structure
- training details (gradient clipping, learning rate schedule, gradeint accumulation)

`computation.md`
- mixed precision

#### implementation
`vanilla-transformer.ipynb`
- next token prediction
- spelled out attention implementation

`model.py`
- GPT-2 implementation
  
`generate_sentence.py`
- top-k sampling for generation

`tiny_train_script.py`
- single epoch training with all the tricks

### <3> modern transformer (Llama)

#### notes
`architecture.md`
- common change in architecture 


## references

https://www.youtube.com/watch?v=zduSFxRajkE

https://www.youtube.com/watch?v=l8pRSuU81PU

https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=140s

https://stanford-cs336.github.io/spring2025/
