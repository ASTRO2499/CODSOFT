def chatbot_response(user_input):
    # Convert the input to lower case to make the matching case-insensitive
    user_input = user_input.lower()

    # Define some predefined responses
    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I help you today?"
    elif 'how are you' in user_input:
        return "I'm just a bot, but I'm doing great! How about you?"
    elif 'bye' in user_input or 'goodbye' in user_input:
        return "Goodbye! Have a nice day!"
    elif 'help' in user_input:
        return "Sure! What do you need help with?"
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
