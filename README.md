# dpt

The idea is to train something similar to gpt-2. However, the resources are
very limited (one gpu). The goal is to create a model that is somewhat capable 
of communicating in natural language with as little knowledge as possible.

I shall call it dpt - dumb pretrained transformer.

Is it possible? What kind of data is needed to train something like that? 
Probably rather books and novels / social media posts instead of wikipedia articles and science literature.

# ToDos

## Data
We need a dataset that we can inspect and curate. Throwing just millions of texts at the problem won't be
possible here. 

## Model
Recreate gpt with the help of karpathy, as well [this repo](https://github.com/karpathy/nanoGPT) as this [video](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=11).

