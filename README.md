
<body>
    <div class="container mt-5">
        <h1 class="text-center">Java Helper Chatbot</h1>
        <h2>Instructions to Run the Project:</h2>
        <ol>
            <li>
                <strong>Clone the Repository:</strong>
                <p>Open your terminal and run the following command to clone the repository:</p>
                <pre><code>git clone https://github.com/yourusername/java-helper-chatbot.git<br>
cd java-helper-chatbot/MultipleFiles</code></pre>
            </li>
            <li>
                <strong>Set Up a Virtual Environment (Optional but Recommended):</strong>
                <p>Create a virtual environment to manage dependencies:</p>
                <pre><code>python -m venv venv<br>
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>
            </li>
            <li>
                <strong>Install Required Packages:</strong>
                <p>Make sure you have Flask, TensorFlow, and nltk installed. You can install them using pip:</p>
                <pre><code>pip install Flask tensorflow nltk</code></pre>
            </li>
            <li>
                <strong>Download NLTK Data:</strong>
                <p>The project uses NLTK for tokenization. Run the following command in Python to download the required data:</p>
                <pre><code>import nltk<br>
nltk.download('punkt')</code></pre>
            </li>
            <li>
                <strong>Run the Flask Application:</strong>
                <p>Start the Flask server by running:</p>
                <pre><code>python app.py</code></pre>
            </li>
            <li>
                <strong>Access the Chatbot:</strong>
                <p>Open your web browser and go to <a href="http://127.0.0.1:5000/" target="_blank">http://127.0.0.1:5000/</a> to access the Java Helper Chatbot.</p>
            </li>
        </ol>
    </div>
</body>
</html>
