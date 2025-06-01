<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Natasha Rahima

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I didn't expect this product to come early. The packaging was damaged, luckily, it still cover the product inside safely. I waited for so long, but what came to me was unpredictable. I guess I have to buy a new one instead.")
```

Result : 

```
[{'label': 'NEGATIVE', 'score': 0.9964536428451538}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the negative tone in the given sentence. The score of 0.99 indicates a high confidence level in the classification result as the sentence contains words like "I didn't expect", "damaged", "unpredictable" and "buy a new one" which are indicative of a negative sentiment.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "The story follows a young girl who wrote diary in th Amsterdam attic where she and her family hid from the Nazis for two years.",
    candidate_labels=["romance", "biography", "fantasy"],
)
```

Result : 

```
{'sequence': 'The story follows a young girl who wrote diary in th Amsterdam attic where she and her family hid from the Nazis for two years.',
 'labels': ['biography', 'romance', 'fantasy'],
 'scores': [0.8068510293960571, 0.09925013035535812, 0.09389881044626236]}
```

Analysis on example 2 : 

The zero-shot classifier accurately identifies "biography" as the most relevant label with a high confidence score (~0.81). This result demonstrates the model's capability to infer contextual meaning from historical and personal storytelling, even without explicit keyword matches.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generators = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generators(
    "This book is about",
    max_length=25, # you can change this
    num_return_sequences=1, # and this too
)
```

Result : 

```
[{'generated_text': 'This book is about a new generation of men, women and children who are seeking to get a voice, to get a voice, to make a voice.”'}]
```

Analysis on example 3 : 

The text generation model produces a coherent and contextually relevant continuation from the prompt "This book is about". While there’s some repetition in the phrase “to get a voice,” the output still conveys a meaningful idea. It reflecs the model’s capacity to construct meaningful and grammatically correct sentences.

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Reading a good <mask> can widen your knowledge.", top_k=4)
```

Result : 

```
[{'score': 0.481833815574646,
  'token': 1040,
  'token_str': ' book',
  'sequence': 'Reading a good book can widen your knowledge.'},
 {'score': 0.09412383288145065,
  'token': 31046,
  'token_str': ' textbook',
  'sequence': 'Reading a good textbook can widen your knowledge.'},
 {'score': 0.08615956455469131,
  'token': 36451,
  'token_str': ' dictionary',
  'sequence': 'Reading a good dictionary can widen your knowledge.'},
 {'score': 0.027688248082995415,
  'token': 1566,
  'token_str': ' article',
  'sequence': 'Reading a good article can widen your knowledge.'}]
```

Analysis on example 3.5 : 

The model thinks "book" is the best word to fill in the blank, and it makes the sentence sound natural. It shows the model's understanding of different reading materials that can contribute to knowledge.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("Haruki Murakami a writer born in Japan.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.97308534),
  'word': 'Haruki Murakami',
  'start': 0,
  'end': 15},
 {'entity_group': 'LOC',
  'score': np.float32(0.999637),
  'word': 'Japan',
  'start': 33,
  'end': 38}]
```

Analysis on example 4 : 

The NER model correctly identifies "Haruki Murakami" as a person (PER) and "Japan" as a location (LOC). This shows the model is good at finding names and places in a sentence.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the most popular book written by George Orwell?"
context = "George Orwell was a British writer known for his critical and dystopian novels. His most popular book is 'Animal Farm', a novel that explores a totalitarian regime and has become a classic in political literature."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.772711455821991, 'start': 106, 'end': 117, 'answer': 'Animal Farm'}
```

Analysis on example 5 : 

The question-answering model successfully finds the answer "Animal Farm" from the context, showing it's effective at picking out specific facts from a given passage. The high confidence score also means the model is quite sure about its answer.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Dead Poets Society is a 1989 American drama film directed by Peter Weir and starring Robin Williams. Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher, John Keating, who inspires his students through his teaching of poetry. Keating's unorthodox methods, including encouraging students to "seize the day" and view life from different perspectives, challenge the traditional values of the school and its administration.
The film explores themes such as conformity, individualism, the power of poetry, and the struggle between passion and duty. As the students form the secret "Dead Poets Society" to appreciate poetry and self-expression, they begin to discover their own voices, but not without consequences. The film ultimately delivers a powerful message about the importance of thinking for oneself and the transformative power of literature and education.
Dead Poets Society was a critical and commercial success, earning an Academy Award for Best Original Screenplay and nominations for Best Picture and Best Actor. It remains a beloved film that continues to inspire audiences around the world.
"""
)
```

Result : 

```
[{'summary_text': ' The film explores themes such as individualism, the power of poetry, and the struggle between passion and duty . The film was nominated for Best Picture and Best Actor . It was a successful film that continues to inspire audiences around the world . It is based on the story of an English teacher at an elite boarding school in Vermont .'}]

```

Analysis on example 6 :

The summarization model captures the main ideas of Dead Poets Society, like its key themes, the school setting, and its success. The summary is shorter but still gives a good idea of what the film is about.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Warna kesukaanku adalah hitam.")
```

Result : 

```
[{'translation_text': 'Ma couleur préférée est noire.'}]

```

Analysis on example 7 :

The translation model accurately translates the Indonesian sentence into French. The meaning is preserved accurately, showing the model works well for basic personal statements.

---

## Analysis on this project

This project shows how different NLP models from Hugging Face Transformers can handle a variety of language tasks. Each example demonstrates that pre-trained models can understand and process text effectively. Overall, this project highlights the usefulness and flexibility of NLP tools in making sense of human language.