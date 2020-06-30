# Export bert model to ONNX, start with example from:
#
#    https://modelzoo.co/model/pytorch-pretrained-bert

import torch
from pytorch_pretrained_bert import BertTokenizer,BertModel, BertForMaskedLM, BertForNextSentencePrediction

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with 'BertForMaskedLM'
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)
assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated with 1st and 2nd sentences (see paper)
segments_ids = [0,0,0,0,0,1,1,1,1,1,1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
#model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-cased')
model.eval()

# dump an onnx file - use the actual inputs since not quite sure how to get integer outputs...
torch.onnx.export(model,(tokens_tensor,segments_tensors),'bertmodel_lm.onnx', verbose=True)

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0,masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print('Predicted token = {}'.format(predicted_token[0]))
assert predicted_token[0] == 'henson'
