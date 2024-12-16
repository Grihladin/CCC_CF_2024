from flask import Flask, request, render_template
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

'''

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

    
    return render_template(
        "homepage.html",
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

    return render_template("result.html",
                                learn=learn, 
                                problem=problem, 
                                output=formated_output, 
                                top_courses=top_courses)

if __name__ == '__main__':
    app.run(debug=True)
