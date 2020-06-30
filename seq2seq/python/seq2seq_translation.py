from io import open
import string
import re
import random
import sys
import migraphx
import numpy as np

SOS_token = 0
EOS_token = 1
max_sent_len = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0 : "SOS", 1 : "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indicesFromSentence(lang, sent):
    return [lang.word2index[word] for word in sent.split(' ')]

def readLangs(lang1, lang2, reverse = True):
    lines = open('../data/%s-%s_procd.txt'%(lang1, lang2)).read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def generate_arg(arg, target):
    if target == "gpu":
        return migraphx.to_gpu(migraphx.argument(arg))
    else:
        return migraphx.argument(arg)

def evaluate(encoder, decoder, input_lang, output_lang, hidden_size, max_sent_len, sent, target_name):

    input_indices = indicesFromSentence(input_lang, sent)
    input_len = len(input_indices)
    encoder_hidden = [float(0)] * hidden_size
    encoder_output = []
    decoder_words = []

    for i in range(input_len):
        encoder_params = {}
        args = []
        result = []
        for key, value in encoder.get_parameter_shapes().items():
            if key == "input.1":
                index = []
                index.append(input_indices[i])
                args.append(np.array(index, dtype = np.longlong).reshape(value.lens()))
            elif key == "hidden":
                args.append(np.array(encoder_hidden).astype(np.single).reshape(value.lens()))
            else:
                hash_val = hash(key) % (2 ** 32 - 1)
                np.random.seed(hash_val)
                args.append(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))

            encoder_params[key]  = migraphx.argument(args[-1])
    
        prog_out = encoder.run(encoder_params)
        cur_output = np.array(prog_out[0]).reshape(-1).tolist()
        encoder_output.extend(cur_output)
        encoder_hidden = np.array(prog_out[1]).reshape(-1).tolist()

    # expected size of the encoder output is hidden_size * max_sent_len
    if hidden_size * max_sent_len > len(encoder_output):
        encoder_output.extend([0] * (hidden_size * max_sent_len - len(encoder_output)))

    # runt the decoder
    decoder_input = [SOS_token]
    decoder_hidden = encoder_hidden
    decoder_words = []

    decoder_output = []
    for i in range(max_sent_len):
        decoder_params = {}
        args = []
        result = []
        for key, value in decoder.get_parameter_shapes().items():
            if key == "input.1":
                args.append(np.array(decoder_input).astype(np.longlong).reshape(()))
            elif key == "hidden":
                args.append(np.array(decoder_hidden).astype(np.single).reshape(value.lens()))
            elif key == "2":
                args.append(np.array(encoder_output).astype(np.single).reshape(value.lens()))
            else:
                hash_val = hash(key) % (2 ** 32 - 1)
                np.random.seed(hash_val)
                args.append(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))

            decoder_params[key] = migraphx.argument(args[-1])

        prog_out = decoder.run(decoder_params)
        decoder_output = np.array(prog_out[0]).tolist()
        decoder_hidden = np.array(prog_out[1]).tolist()

        index = np.argmax(decoder_output)
        
        if index == EOS_token:
            break
        else:
            decoder_words.append(output_lang.index2word[index])
            decoder_input[0] = index

    return decoder_words

def main():
    if len(sys.argv) != 6:
        print("Usage: python seq2seq_translation.py encoder.onnx decoder.onnx in_lang out_lang gpu/cpu")
        exit()


    # read language file and create dictionary
    input_lang, output_lang, pairs = readLangs('eng', 'fra', False)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    # load onnx files of encoder and decoder
    encoder = migraphx.parse_onnx(sys.argv[1])
    decoder = migraphx.parse_onnx(sys.argv[2])
    hidden_size = 256

    target_name = "cpu"
    if sys.argv[5] == "gpu":
        target_name = "gpu"

    encoder.compile(migraphx.get_target(target_name))
    decoder.compile(migraphx.get_target(target_name))

    sent_num = 500
    for sent_no in range(sent_num):
        print("sent_no = {}".format(sent_no))
        sent_index = sent_no * 10
        print("> {}".format(pairs[sent_index][0]))
        print("= {}".format(pairs[sent_index][1]))
        out_words = evaluate(encoder, decoder, input_lang, output_lang, 
            hidden_size, max_sent_len, pairs[sent_index][0], target_name)
        out_sent = ' '.join(out_words)
        print('<', out_sent)
        print('')

if __name__ == "__main__":
    main()

