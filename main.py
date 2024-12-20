from flask import Flask, request, render_template, redirect, url_for
import ollama
import chromadb
import json
import markdown
import os

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

@app.route('/', methods=['GET', 'POST'])
def user_questions():
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
            'options': ['Beginner', 'Medium', 'Advanced']
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

    
    return render_template(
        "homepage.html",
        question=questions[current_step],
        current_step=current_step,
        total_steps=len(questions),
        saved_data=saved_data
    )

def process_submission(saved_data):
    problem, learn, knowledge_level, department = saved_data
    
    prompt = f"I face the problem {problem} and I want to {learn}, my current knowledge level is {knowledge_level}"

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

    return render_template("result.html",
                                learn=learn, 
                                problem=problem,
                                knowledge_level = knowledge_level,
                                final_prompt = prompt,
                                output=formated_output, 
                                top_courses=top_courses)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    rating = request.form['rating']
    feedback_text = request.form['feedback']
    feedback = {
        'rating': rating,
        'feedback': feedback_text
    }

    feedback_file = 'feedback.json'

    if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []

    feedback_data.append(feedback)

    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=4)

    return redirect(url_for('user_questions'))

@app.route('/pro_mode', methods=['POST'])
def pro_mode():
    prompt = request.form['proQuestion']
    
    response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=prompt)
    embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=30
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

    return render_template("proResult.html", 
                                search_prompt=prompt,
                                top_courses=top_courses)

if __name__ == '__main__':
    app.run(debug=True)
