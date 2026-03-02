from ollama import generate

while True:
    query = input("\nYou: ")

    if query.lower() == exit:
        break

    reponse = generate(model="llama3.2:3b", prompt=query, stream=True)

    print("\n🚀 Llama is thinking...")
    print(f"\nAI: ", end="", flush=True)
    for chunk in reponse:
        print(chunk["response"], end="", flush=True)
