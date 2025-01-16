# d-p-t

The idea is to train something similar to gpt-2. However, the resources are
very limited (one gpu). The goal is to create a model that is somewhat capable 
of communicating in natural language with as little knowledge as possible.

I shall call it d-p-t: dumb pretrained transformer.

Is it possible? What kind of data is needed to train something like that? 
Probably rather books and novels / social media posts instead of wikipedia articles and science literature.

# ToDos
* proper templating for instruction finetuning

## Data
We need a dataset that we can inspect and curate. Throwing just millions of texts at the problem won't be
possible here. 
As starter, I take the [fineweb-edu-fortified](https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified) dataset. Reads like higher quality after deduplicating. I also filtered the data to get
only a score of 3 and higher.

## Model
Recreate gpt with the help of karpathy, as well [this repo](https://github.com/karpathy/nanoGPT) as this [video](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=11).

