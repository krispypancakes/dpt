when the mlp is where knowledge is stored and the attention parts are the knowledge transfer,
could it make sense to scale down the mlp dims if we want a massively dumb model ?

the size comes from the vocab size: 50257 and the vector length: 768
-> probably use a simpler tokenizer as well so we have smaller matrices

context or sequence length does also play a big role.


TODO:
- check when what is copied to gpu and if that makes sense
