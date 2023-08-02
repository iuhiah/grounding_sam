with open(r"arguments.txt") as f:
    text = f.read().splitlines()
    prompts = [i.strip() for i in text[1].split(",")]

    print(text)