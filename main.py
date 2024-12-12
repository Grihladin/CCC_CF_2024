from flask import Flask, request, render_template_string
import ollama
import chromadb
import json
import markdown

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
    <style>
        /* Modern Base Styles */
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        /* Elegant Title Header */
        .title-header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 40px auto;
            max-width: 600px;
            padding: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* Modern Form Container */
        .form-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto;
            transition: transform 0.3s ease;
        }

        .form-container:hover {
            transform: translateY(-5px);
        }

        /* Enhanced Form Elements */
        .form-container h1 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        .form-control {
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 0.2rem rgba(52,152,219,0.25);
        }

        /* Modern Button Style */
        .btn-primary {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52,152,219,0.4);
        }

        /* Result Styles */
        .accordion .card {
            border: none;
            margin-bottom: 10px;
            border-radius: 8px;
            overflow: hidden;
        }

        .accordion .card-header {
            background: #f8f9fa;
            border: none;
            padding: 0;
        }

        .accordion .btn-link {
            width: 100%;
            text-align: left;
            color: #2c3e50;
            font-weight: 600;
            text-decoration: none;
            padding: 15px 20px;
        }

        .accordion .card-body {
            padding: 20px;
            background: #fff;
        }

        /* Output Text Styling */
        .output-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .title-header {
                font-size: 2rem;
                padding: 15px;
            }

            .form-container {
                margin: 20px;
                padding: 20px;
            }
        }

        /* Form Group Sizing */
        .form-group {
            margin-bottom: 2rem; 
            padding: 1rem;
        }

        .form-control {
            min-height: 3rem;
            font-size: 1.1rem;
            padding: 1rem;
        }

        textarea.form-control {
            min-height: 150px;
        }

        /* For select dropdowns */
        select.form-control {
            height: 3rem;
            padding: 0 1rem;
        }
    </style>
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

    <strong class="mt-4 d-block">Learning assistant:</strong>
    <div class="output-text">{{ output | safe }}</div>
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

@app.route('/', methods=['GET', 'POST'])
def index():
    questions = [
        {
            'id': 'problem',
            'label': 'What is your problem:',
            'placeholder': 'Explain in your own words what task you want to do. What isn\'t working the way you want it to? What should be improved or changed?'
        },
        {
            'id': 'learn',
            'label': 'What you want to learn:',
            'placeholder': 'What information you need to solve your problem. What are you trying to understand?'
        },
        {
            'id': 'knowledge_level',
            'label': 'What is your current knowledge level:',
            'type': 'select',
            'options': ['Beginner', 'Medium', 'Advanced']  # Fixed typo in 'Beginner'
        },
        {
            'id': 'department',
            'label': 'Choose your department:',
            'type': 'select',
            'options': ['German Department', 'China Department', 'USA Department', 'UK Department', 
                       'France Department', 'Italy Department', 'Spain Department', 'Japan Department', 
                       'India Department', 'Brazil Department', 'Russia Department', 'Canada Department']
        }
    ]

    current_step = int(request.form.get('step', 0))
    saved_data = request.form.getlist('saved_data[]')

    if request.method == 'POST':
        current_answer = request.form.get(questions[current_step]['id'])
        saved_data.append(current_answer)
        current_step += 1

        if current_step >= len(questions):
            # Process final submission
            return process_submission(saved_data)

    template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Meta Tags -->
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daimler Truck LMS</title>
    
    <!-- External CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    
    <!-- Custom Styles -->
    <style>
        /* Modern Base Styles */
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        /* Elegant Title Header */
        .title-header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 40px auto;
            max-width: 600px;
            padding: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* Modern Form Container */
        .form-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto;
            transition: transform 0.3s ease;
        }

        .form-container:hover {
            transform: translateY(-5px);
        }

        /* Enhanced Form Elements */
        .form-container h1 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        .form-control {
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 0.2rem rgba(52,152,219,0.25);
        }

        /* Modern Button Style */
        .btn-primary {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52,152,219,0.4);
        }

        /* Result Styles */
        .accordion .card {
            border: none;
            margin-bottom: 10px;
            border-radius: 8px;
            overflow: hidden;
        }

        .accordion .card-header {
            background: #f8f9fa;
            border: none;
            padding: 0;
        }

        .accordion .btn-link {
            width: 100%;
            text-align: left;
            color: #2c3e50;
            font-weight: 600;
            text-decoration: none;
            padding: 15px 20px;
        }

        .accordion .card-body {
            padding: 20px;
            background: #fff;
        }

        /* Output Text Styling */
        .output-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .title-header {
                font-size: 2rem;
                padding: 15px;
            }

            .form-container {
                margin: 20px;
                padding: 20px;
            }
        }

        /* Form Group Sizing */
        .form-group {
            margin-bottom: 2rem;  /* Increased bottom margin */
            padding: 1rem;        /* Added padding around inputs */
        }

        .form-control {
            min-height: 3rem;     /* Increased minimum height */
            font-size: 1.1rem;    /* Slightly larger font */
            padding: 1rem;        /* More padding inside inputs */
        }

        textarea.form-control {
            min-height: 150px;    /* Taller textareas */
        }

        /* For select dropdowns */
        select.form-control {
            height: 3rem;         /* Match other inputs */
            padding: 0 1rem;      /* Horizontal padding */
        }
    </style>
</head>

<body>
    <!-- Page Header -->
    <h1 class="title-header">Daimler Truck LMS</h1>

    <!-- Main Content -->
    <div class="container">
        <div class="form-container">
            <!-- Form Header -->
            <h1>{{ question.label }}</h1>

            <!-- Question Form -->
            <form action="/" method="post">
                <div class="form-group">
                    {% if question.type == 'select' %}
                        <!-- Department Selector -->
                        <select class="form-control" 
                                id="{{ question.id }}" 
                                name="{{ question.id }}">
                            {% for option in question.options %}
                                <option>{{ option }}</option>
                            {% endfor %}
                        </select>
                    {% else %}
                        <!-- Text Input -->
                        <textarea class="form-control" 
                                  id="{{ question.id }}" 
                                  name="{{ question.id }}" 
                                  placeholder="{{ question.placeholder }}"
                                  rows="4"></textarea>
                    {% endif %}
                </div>

                <!-- Hidden Fields -->
                <input type="hidden" name="step" value="{{ current_step }}">
                {% for data in saved_data %}
                    <input type="hidden" name="saved_data[]" value="{{ data }}">
                {% endfor %}

                <!-- Submit Button -->
                <button type="submit" class="btn btn-primary">
                    {{ 'Submit' if current_step == 3 else 'Next step' }}
                </button>
            </form>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
    '''
    
    return render_template_string(
        template,
        question=questions[current_step],
        current_step=current_step,
        saved_data=saved_data
    )

def process_submission(saved_data):
    problem, learn, knowledge_level, department = saved_data
    
    prompt = f"I face the problem {problem} and I want to {learn}"

    # User prompt embedding
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
        prompt=f"""As a professional learning consultant, analyze the data {top_courses} and provide a response to this request: {prompt}.
        Build your answer only on provided data, use the given information to make a clear, helpful recommendation while maintaining a polite and supportive tone."""
    )
    ollama_answer = output['response']
    formated_output = markdown.markdown(ollama_answer)

    return render_template_string(result_template, 
                                learn=learn, 
                                problem=problem, 
                                output=formated_output, 
                                top_courses=top_courses)

if __name__ == '__main__':
    app.run(debug=True)
