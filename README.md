#NLP Project 2023

## FNC-I: STANCE DETECTION.
Our topic is [Fake News Challenge Stage 1 (FNC-I)](http://www.fakenewschallenge.org/): Stance Detection Task.

Assessing the veracity of a news story is a complex task even for trained experts. Fortunately, the process can be broken down into steps. A helpful first step towards identifying fake news is to understand what other news organizations are saying about the topic. Automating this process, called Stance Detection, could serve as a useful building block in an AI-assisted fact-checking pipeline. So stage #1 of the Fake News Challenge (FNC-1)focuses on the task of Stance Detection.
 
## Report
Our [report](https://github.com/BiteKirby3/FakeNewsChallenge/blob/main/report.pdf) can be found in the root of this repository.

## Code
We explored different approaches throughout this project. To check our code, go to the [code](https://github.com/BiteKirby3/FakeNewsChallenge/tree/main/code) folder.

## Bonus
Our finest model, achieved through the process of fine-tuning RoBERTa, stands out as a significant advancement compared to the winning solution of the FNC-I challenge. You can readily employ our fine-tuned RoBERTa model to perform accurate and reliable stance detection on news articles. The utilization of our model for news stance inference is straightforward and can be implemented as follows: 

1. Install the Hugging Face's [transformers](https://huggingface.co/docs/transformers/v4.17.0/en/index) library
```pip install transformers```

2. Import the essential modules for tokenization and model loading
```
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
```

3. Infer the stance!
```
def classify_stance(headline, body):
  model = AutoModelForSequenceClassification.from_pretrained('sxie3333/RoBERTa')
  tokenizer = AutoTokenizer.from_pretrained('sxie3333/RoBERTa')
  inputs = tokenizer.__call__(headline, body, return_tensors="pt", truncation=True,padding=True,max_length=512)
  
  with torch.no_grad():
    logits = model(**inputs).logits
	
  #get the class with the highest probability
  predicted_class_id = logits.argmax().item()

  return model.config.id2label[predicted_class_id]
```