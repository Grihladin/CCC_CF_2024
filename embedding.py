import pandas as pd
import ollama
import chromadb
import json

# Load the data
csv_file = "Coursera.csv"
df = pd.read_csv(csv_file)

# Extract data i need from datas
df['combined'] = df['Course Name'] + ": " + df['Course Description'] + ": " + df['Skills']
courses = df['combined'].tolist()

client = chromadb.Client()
collection = client.create_collection(name="docs")

# Add the preprocessed embeddings to the collection
embeddings_data = []
for i, course in enumerate(courses):
    response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=course)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[course]
    )
    embeddings_data.append({
        "id": str(i),
        "embedding": embedding,
        "document": course
    })

# Save the embeddings and documents to a file
with open("embeddings_data.json", "w") as f:
    json.dump(embeddings_data, f)
