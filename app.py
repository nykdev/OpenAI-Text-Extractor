from flask import Flask, request, jsonify
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Set your OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
prompt = os.getenv("OPENAI_PROMPT")
client = OpenAI(api_key=api_key)

# Function to process base64 encoded image with OpenAI GPT-4o and extract required details
def process_image_with_openai(image_url):
    GPT_MODEL = "gpt-4o"
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user","content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": image_url}
                    }
                ]
            }
        ],
        temperature=0.5,
    )
    print(response)
    response_content = response.choices[0].message.content
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
    if json_match:
        response_json = json_match.group(1)
        return response_json
    else:
        return None

# API endpoint to process image URL and extract details with GPT-4o
@app.route('/extract-details', methods=['POST'])
def extract_details():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    image_url = data['image_url']

    try:
        # Process the image URL with GPT-3.5 Turbo
        processed_text = process_image_with_openai(image_url)

        if processed_text:
            return jsonify({'processed_text': processed_text}), 200
        else:
            return jsonify({'error': 'Failed to extract JSON from response'}), 500
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5123, debug=True)
