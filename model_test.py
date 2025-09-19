# Save this as test_qwen3_maze.py
# Run with: python test_qwen3_maze.py

from mlx_lm import load, generate, stream_generate


def main():
    model_id = "nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx"

    # Load model + tokenizer
    model, tokenizer = load(model_id)

    # Role-based chat prompt
    messages = [
        {
            "role": "system",
            "content": "You are a puzzle and maze designer. You generate ASCII mazes that are solvable, with clear start and end points."
        },
        {
            "role": "user",
            "content": (
                # "Generate a maze in ASCII art format. The Maze should be solvable. the should should be between 5x5 to 10x10."
                "Generate a maze in ASCII art format. The maze must follow these rules:\n\n"
                "1. Represent the maze as a grid using characters:\n"
                "   - '#' for walls\n"
                "   - ' ' (space) for open paths\n"
                "   - 'S' for the start location\n"
                "   - 'E' for the exit (end) location\n\n"
                "2. The maze should be at least 15x15 characters in size.\n\n"
                "3. There must be exactly one 'S' and one 'E', placed on different edges of the maze.\n\n"
                "4. The maze must be solvable: there should be at least one clear path connecting 'S' to 'E'.\n\n"
                "5. Do not include any solution path markings in the maze itself, only the maze.\n\n"
                "After generating the maze, verify its solvability and ensure the output is only the ASCII maze (no explanation)."
                "Quick generation is required dont think to much"
            )
        }
    ]

    # Apply chat template for Qwen models
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate maze
    output = generate(
    # stream = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=9000
    )

    print("=== Maze Output ===")
    print(output.strip())

    # collected = []
    # for token in stream:
    #     print(token, end="", flush=True)
    #     collected.append(token)
    #
    # print("\n\n=== Generation Finished ===")

    # If you want the full maze string:
    # full_output = "".join(collected).strip()
    # Optionally: save to file
    # with open("maze.txt", "w") as f:
    #     f.write(full_output)


if __name__ == "__main__":
    main()
