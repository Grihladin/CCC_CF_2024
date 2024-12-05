import pandas as pd
import ollama
import chromadb
import json

csv_file = "Coursera.csv"
df = pd.read_csv(csv_file)

# Extract data needed from DataFrame
df['combined'] = df['Course Name'] + ": " + df['Course Description'] + ": " + df['Skills']

courses = []
for index, row in df.iterrows():
    course_name = row['Course Name']
    combined_text = row['combined']
    courses.append({
        'name': course_name,
        'text': combined_text
    })

client = chromadb.Client()
collection = client.create_collection(name="docs")

embeddings_data = []
for i, course in enumerate(courses):
    response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=course['text'])
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[course['text']]
    )
    embeddings_data.append({
        "id": str(i),
        "name": course['name'],
        "embedding": embedding,
        "document": course['text']
    })

# Save embeddings data with course names
with open("embeddings_data.json", "w") as f:
    json.dump(embeddings_data, f)

