from llm import LLMEngine

def main():
    print("What kind of game would you like to play? Briefly describe it.")
    user_game_description = input()
    print("Generating...\n\n")
    engine = LLMEngine(user_game_description)

    print(engine.step())
    while True:
        command = input()
        print(engine.step(command))

if __name__ == "__main__":
    main()
