import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']
print('tokenized_tex = {}'.format(tokenized_text))


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print('indexed_tokens = {}'.format(indexed_tokens))
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print('tokens_tensor = {}'.format(tokens_tensor))
print('segments_tensor = {}'.format(segments_tensors))

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

encoded_layers, _ = model(tokens_tensor, segments_tensors)

print('encoded_layers = {}'.format(len(encoded_layers)))

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

predictions = model(tokens_tensor, segments_tensors)

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

print('predicted_token = {}'.format(predicted_token))

