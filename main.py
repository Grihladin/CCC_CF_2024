from flask import Flask, request, render_template_string
import ollama
import chromadb
import json



app = Flask(__name__)

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
    <title>Result</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <h1>Result</h1>
        <p><strong>Problem:</strong> {{ problem }}</p>
        <p><strong>Learn:</strong> {{ learn }}</p>

        <h3>Top 5 Courses:</h3>
        <div class="accordion" id="courseAccordion">
            {% for course in top_courses %}
                <div class="card">
                    <div class="card-header" id="heading{{ loop.index }}">
                        <h2 class="mb-0">
                            <button class="btn btn-link" type="button" data-toggle="collapse"
                                    data-target="#collapse{{ loop.index }}" aria-expanded="false" aria-controls="collapse{{ loop.index }}">
                                {{ course.name }}
                            </button>
                        </h2>
                    </div>
                    <div id="collapse{{ loop.index }}" class="collapse"
                         aria-labelledby="heading{{ loop.index }}" data-parent="#courseAccordion">
                        <div class="card-body">
                            {{ course.description }}
                        </div>
                    </div>
                </div>
            {% endfor %}
        </div>

        <p><strong>Learning assistant:</strong> {{ output }}</p>
    </div>

    <!-- Bootstrap JS and dependencies -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <!-- Popper.js -->
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <!-- Bootstrap JS -->
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
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
        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    '''

@app.route('/submit', methods=['POST'])
def submit():
    learn = request.form['learn']
    problem = request.form['problem']
    department = request.form['department']

    prompt = f"I face the problem {problem} and I want to learn {learn}"

    # User promt embedding
    response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=prompt)
    embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    # Get the IDs of the top documents
    top_ids = results['ids'][0]
    id_to_course = {item['id']: {'name': item['name'], 'description': item['document']} for item in embeddings_data}

    top_courses = []
    for doc_id in top_ids:
        course_info = id_to_course[doc_id]
        top_courses.append({
            'name': course_info['name'],
            'description': course_info['description']
        })


    output = ollama.generate(
        model="llama3.1:latest",
        prompt=f""" Imagine you are a learning consultant, Using this data {top_courses}. 
        Respond to this prompt, be polite and friendley try match the learning request with problem. Build your answer only on the base of provided data. {prompt}"""
    )
    ollama_answer = output['response']

    return render_template_string(result_template, learn=learn, problem=problem, output=ollama_answer, top_courses=top_courses)

if __name__ == '__main__':
    app.run(debug=True)
