import torch
from markupsafe import escape
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("0xsuid/simba-125M")
model.resize_token_embeddings(len(tokenizer))

def format_input(input_problem):
        answer_type = "\nUse Standard Input format\n"
        formatted_input = "\nQUESTION:\n" + input_problem + "\n" + answer_type + "\nANSWER:\n"
        return formatted_input

# Get a prediction
def get_prediction(input_problem):
    formatted_input = format_input(input_problem)
    encoded_text = tokenizer.encode(formatted_input, truncation=True)
    input_ids = torch.LongTensor(encoded_text).unsqueeze(0)
    output_ids = model.generate(
        input_ids,
        num_beams=5,
        early_stopping=True,
        max_length=2048 - len(input_ids)
    )

    prediction = tokenizer.decode(output_ids[0],skip_special_tokens=True)
    return escape(prediction)

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        coding_problem = request.form.get('coding_problem')
        if coding_problem is not None:
            prediction = get_prediction(coding_problem)
            return render_template('index.html', generatedAnswer=prediction)


if __name__ == '__main__':
    app.run()