import json

import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from spacy.training import offsets_to_biluo_tags

import csv

# Function to load and preprocess the data
def load_data(file_path, include_id=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    formatted_data = []
    for entry in data:
        text = entry['data']['content']
        entities = []
        for annotation in entry['annotations']:
            for result in annotation.get('result', []):
                start = result['value']['start']
                end = result['value']['end']
                labels = result['value']['labels']
                for label in labels:
                    entities.append((start, end, label))
        formatted_data.append((text, {"entities": entities}, entry['id'] if include_id else None))

    return formatted_data

# Function to adjust entity boundaries
def adjust_entity_boundaries(doc, entities):
    adjusted_entities = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            adjusted_entities.append((span.start_char, span.end_char, label))
    return adjusted_entities


# Function to predict entities in a given text
def predict_entities(model, text):
    doc = model(text)
    predictions = []
    for ent in doc.ents:
        predictions.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    return predictions

spacy.require_gpu()

# Define the configuration for the transformer
config = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v3",
        "name": "roberta-base",
        "tokenizer_config": {"use_fast": True},
        "get_spans": {"@span_getters": "spacy-transformers.strided_spans.v1", "window": 512, "stride": 256}
    }
}

# Load SpaCy's blank English model
nlp = spacy.blank("en")

# Load training data
training_file_path = 'C:/Users/haojiecao/Documents/UGS/ner/project-5-at-2023-12-15-22-24-6ca1feed.json'  # Replace with your file path for training
training_data = load_data(training_file_path)

# Load prediction data
prediction_file_path = 'C:/Users/haojiecao/Documents/UGS/ner/project-5-at-2023-12-15-22-24-6ca1feed.json'  # Replace with your file path for prediction
prediction_data = load_data(prediction_file_path, include_id=True)

adjusted_training_data = []
for text, annotations, entry_id in training_data:
    doc = nlp.make_doc(text)
    adjusted_entities = adjust_entity_boundaries(doc, annotations['entities'])
    adjusted_training_data.append((text, {"entities": adjusted_entities}, entry_id))

# Add a transformer model to the pipeline
nlp.add_pipe("transformer", name="roberta-transformer", config=config)

# Add a NER component and add labels
ner = nlp.add_pipe("ner")
for _, annotations, _ in adjusted_training_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Initialize the model
nlp.initialize()

# Training loop
NUM_EPOCHS = 60
BATCH_SIZE = 4
DROPOUT = 0.1

for epoch in range(NUM_EPOCHS):
    losses = {}
    batches = minibatch(adjusted_training_data, size=BATCH_SIZE)
    for batch in batches:
        texts, annotations, _ = zip(*batch)
        examples = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
        nlp.update(examples, drop=DROPOUT, losses=losses)
    print(f"Losses at epoch {epoch}: {losses}")

# Predict and save the results
with open('C:/Users/haojiecao/Documents/UGS/ner/predicted_entities_02.csv', mode='w', newline='',
          encoding='utf-8') as file:
    writer = csv.writer(file)
    for text, _, entry_id in prediction_data:
        predicted_entities = predict_entities(nlp, text)
        for entity in predicted_entities:
            writer.writerow([entry_id, entity[3], entity[0]])  # entry_id, label, text segment

# Save the model
model_dir = "C:/Users/haojiecao/Documents/UGS/ner/spacy_model_02"
nlp.to_disk(model_dir)