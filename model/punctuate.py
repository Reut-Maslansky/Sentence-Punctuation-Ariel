import torch
import numpy as np
import pickle

def punctuate(test_sentence):
    punc = {",": "comma", ".": "period", "?": "question_mark", ":": "colon", ";": "semicolon", "_": "dash",
            "-": "hyphen", ")": "right_round_bracket", "(": "left_round_bracket", "]": "right_square_bracket",
            "[": "left_square_bracket", "\\": "slash", "'": "apostrophe", "\"": "speech_mark"}

    model = torch.load("model/punctuation_model.pth", map_location=torch.device('cpu'))
    model.eval()
    tokenizer = torch.load("model/tokenizer.pth")
    # tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
    tag_values = torch.load("model/tag_values.pth")
    # tag_values = pickle.load(open("model/tag_values.pkl", "rb"))

    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])#.cuda()

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)


    for token, label in zip(new_tokens, new_labels):
        print("label: {}\ttoken: {}".format(label, token))

    # result = ""
    # val_list = list(punc.values())
    # key_list = list(punc.keys())
    # for token, label in zip(new_tokens, new_labels):
    #     print("{}\t{}".format(label, token))
    #     result = result + token
    #     if (label != 'O'):
    #         result = result + key_list[val_list.index(label)]
    #     result = result + " "
    # print(result)
    # return result
    return test_sentence


