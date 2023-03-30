

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# from optimum.bettertransformer import BetterTransformer

import time


# Loading the model and tokenizer for bart-large-cnn

tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model = model.to(0)
# model = BetterTransformer.transform(model)

original_text = 'A home, or domicile, is a space used as a permanent or semi-permanent residence for one or many humans, and sometimes various companion animals. It is a fully or semi sheltered space and can have both interior and exterior aspects to it. Homes provide sheltered spaces, for instance rooms, where domestic activity can be performed such as sleeping, preparing food, eating and hygiene as well as providing spaces for work and leisure such as remote working, studying and playing.'

# Encoding the inputs and passing them to model.generate()
# inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt', max_length = 20, truncation=True)

for i in range(5):
    st = time.time()
    inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt')
    inputs.to(0)
    summary_ids = model.generate(inputs['input_ids'], early_stopping=True)

    # Decoding and printing the summary
    bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(bart_summary)
    print (time.time()-st)





