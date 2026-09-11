from pathlib import Path

def load_prompt(prompt_name: str)-> str:
    prompt_dir = Path("prompts")
    prompt_file = prompt_dir / f'{prompt_name}.txt'
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt