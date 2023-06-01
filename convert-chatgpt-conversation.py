import os
import json


class Message:
    def __init__(self, role, content, create_time, model=None):
        self.role = role
        self.content = content
        self.create_time = create_time
        self.model = model


class Discussion:
    def __init__(self, messages, create_time, title):
        self.messages = messages
        self.create_time = create_time
        self.title = title


json_dir = '.'

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

question_answer_pairs = []

for file in json_files:
    with open(os.path.join(json_dir, file), 'r') as f:
        discussions = json.load(f)

        # Extract the user and assistant messages into an array of questions and answers
        for discussion in discussions:
            question = ""
            answer = ""

            for message_data in discussion['messages']:
                message = Message(**message_data)
                # Ensure message and content are defined and content is not empty
                if message and message.content and len(message.content) > 0:
                    if message.role == "user":
                        question = message.content[0]
                    elif message.role == "assistant":
                        answer = message.content[0]

                # Skip this iteration if the question which has more than 300 characters
                if len(question) > 300:
                    question = ""
                    answer = ""
                    continue

                if question and answer:
                    question_answer_pairs.append(
                        {'question': question, 'answer': answer})
                    # Clear question and answer for the next pair
                    question = ""
                    answer = ""

# Save questionAnswerPairs to a JSON file
output_file_path = 'conversations.json'
with open(output_file_path, 'w') as f:
    json.dump(question_answer_pairs, f, indent=2)
