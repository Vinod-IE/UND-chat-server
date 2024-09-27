from flask import Flask, request, jsonify
from flask_cors import CORS
from main import setup
import markdown2  # Library for converting markdown to HTML
import logging

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Initialize query handler
query_handler, responder, memory = setup()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query', '')

        if query.lower() == 'quit':
            return jsonify({"error": "Session ended"}), 400

        # Get the previous interaction context
        memory_context = memory.get_context()

        # Handle the query and get the response
        context = query_handler.handle(query)
        full_context = f"{memory_context}\n\n{context}"
        answer = responder.respond(query, full_context)

        # Convert the answer from Markdown to HTML
        formatted_answer = markdown2.markdown(answer)

        # Save the conversation history in memory
        if "technical difficulties" not in answer and "unable to provide an answer" not in answer:
            memory.add(query, answer)

        # Send the HTML response
        return jsonify({"answer": formatted_answer}), 200
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True,use_reloader=False)
