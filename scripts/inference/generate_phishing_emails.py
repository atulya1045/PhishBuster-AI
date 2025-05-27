from vertexai.preview.language_models import ChatModel

chat_model = ChatModel.from_pretrained("gemini-1.0-pro")  # Correct model name
chat = chat_model.start_chat()
response = chat.send_message("Generate a phishing email about a fake job offer.")
print(response.text)
