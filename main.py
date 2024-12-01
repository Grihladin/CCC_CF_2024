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
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <h1>Result</h1>
        <p><strong>Learn:</strong> {{ learn }}</p>
        <p><strong>Problem:</strong> {{ problem }}</p>
        <h3>Top 5 Courses:</h3>
        <ul class="list-group">
            {% for doc in top_5_documents %}
                <li class="list-group-item">{{ doc }}</li>
            {% endfor %}
        </ul>
        <p><strong>Learning assistant:</strong> {{ output }}</p>
    </div>
    <!-- Bootstrap JS and dependencies (Optional) -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
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
        <title>Let's find best courses</title>
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
            <a class="navbar-brand" href="#">Team7 SolutionDemo</a>
        </nav>
        <div class="container mt-5">
            <h1>Let's find best courses</h1>
            <form action="/submit" method="post">
                <div class="form-group">
                    <label for="problem">What is your problem:</label>
                    <textarea class="form-control" id="problem" name="problem"></textarea>
                </div>
                <div class="form-group">
                    <label for="learn">What you want to learn:</label>
                    <textarea class="form-control" id="learn" name="learn"></textarea>
                </div>
                <div class="form-group">
                    <label for="department">Choose your department:</label>
                    <select class="form-control" id="department" name="department">
                        <option>German Department</option>
                        <option>China Department</option>
                        <option>USA Department</option>
                        <option>UK Department</option>
                        <option>France Department</option>
                        <option>Italy Department</option>
                        <option>Spain Department</option>
                        <option>Japan Department</option>
                        <option>India Department</option>
                        <option>Brazil Department</option>
                        <option>Russia Department</option>
                        <option>Canada Department</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-primary">Submit</button>
            </form>
        </div>
        <!-- Bootstrap JS and dependencies (Optional) -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''
@app.route('/submit', methods=['POST'])

def submit():
    learn = request.form['learn']
    problem = request.form['problem']
    department = request.form['department']

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
