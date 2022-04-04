from flask import Flask, render_template, request
from model.punctuate import punctuate

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        return render_template("index.html", punc=punctuate(text))


if __name__ == '__main__':
    app.run(debug=True)

    import os

    # os.add_dll_directory("C:\Dev\Sentence Punctuation\server\\venv\Lib\site-packages")
    # import torch
    #
    # model = torch.load("model/punctuation_model.pth", map_location=torch.device('cpu'))
    # model.eval()
    #
    # tokenizer = torch.load("model/tokenizer.pth")
    # tokenizer.eval()

    # print(torch.cuda.is_available())

    # def punc(text):
    #     tokenized_sentence = tokenizer.encode(text)
    #     return torch.tensor([tokenized_sentence]).cuda
