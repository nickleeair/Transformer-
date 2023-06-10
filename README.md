# Transformer-
Transformer original model for translate English to France with Bleu score to calculate accuracy 


# Train
`python train.py -load_weights weights -src_lang en_core_web_sm -trg_lang fr_core_news_sm`

# Translate
`python translate.py -load_weights weights -src_lang en_core_web_sm -trg_lang fr_core_news_sm`
