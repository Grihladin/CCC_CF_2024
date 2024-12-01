from flask import Flask, request, render_template_string
import ollama
import chromadb
import json



app = Flask(__name__)

# Load the preprocessed embeddings and documents
with open("embeddings_data.json", "r") as f:
    embeddings_data = json.load(f)

client = chromadb.Client()
collection = client.create_collection(name="docs")

# Add the preprocessed embeddings and documents to the collection
for data in embeddings_data:
    collection.add(
        ids=[data["id"]],
        embeddings=[data["embedding"]],
        documents=[data["document"]]
    )
# HTML template for displaying output
result_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Result</title>
</head>
<body>
    <h1>Result</h1>
    <p>Learn: {{ learn }}</p>
    <p>Problem: {{ problem }}</p>
    <p>Top 5 Courses:</p>
    <ul>
        {% for doc in top_5_documents %}
            <li>{{ doc }}</li>
        {% endfor %}
    </ul>
    <p>Output: {{ output }}</p>
</body>
</html>
'''

@app.route('/')
def index():
    # Serve the form page
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feedback Form</title>
            <style>
                textarea {
                    width: 100%;
                    height: 150px;
                }
            </style>
        </head>
        <body>
            <h1>Team7 SoluthionDemo</h1>
            <form action="/submit" method="post">
                <label for="learn">What you want to learn:</label><br>
                <textarea id="learn" name="learn"></textarea><br><br>
                <label for="problem">What is your problem:</label><br>
                <textarea id="problem" name="problem"></textarea><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
    '''
@app.route('/submit', methods=['POST'])


def submit():
    learn = request.form['learn']
    problem = request.form['problem']

    prompt = "I face the problem" + problem + " and I want to learn" + learn
    response = ollama.embeddings(
        prompt=prompt,
        model="mxbai-embed-large:latest"
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=5
    )

    top_5_documents = [results['documents'][0][i] for i in range(min(5, len(results['documents'][0])))]
    data = " ".join(top_5_documents)
	
    output = ollama.generate(
        model="llama3.1:latest",
        prompt=f""" Imagine you are a learning consultant, Using this data {data}. 
        Respond to this prompt, be polite and friendley try match the learning request with problem. Build your answer only on the base of provided data. {prompt}"""
    )
    python_output = output['response']

    return render_template_string(result_template, learn=learn, problem=problem, top_5_documents=top_5_documents, output=python_output)

if __name__ == '__main__':
    app.run(debug=True)
