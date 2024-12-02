# Team7 SolutionDemo

This project is a web application designed to help users discover the most relevant courses tailored to their specific problems and learning goals. Built using Flask as the web framework, the application integrates advanced document embedding and analysis tools to enhance recommendation quality.

Key features include:
	•	Course Recommendations: Utilizes the “mxbai-embed-large:latest” embedding tool to process and match user queries with suitable course content.
	•	Result Analysis and Summarization: Employs the LLaMA 3.1 model to analyze and provide concise summaries of recommended courses, ensuring users can make informed decisions.

The combination of these technologies delivers an intuitive and efficient platform for personalized learning.

## Features

- User-friendly form to input problems and learning goals
- Dropdown menu to select the department
- Displays top 5 recommended courses based on user input



## Requirements

- Python 3.12
- Installed "mxbai-embed-large:latest" and "llama3.1:latest"
- See `requirements.txt` for the list of required Python packages
