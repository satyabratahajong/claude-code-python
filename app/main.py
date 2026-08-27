import argparse
import json
import os
import sys

from openai import OpenAI


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
]


def execute_tool(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "Read":
        file_path = arguments["file_path"]

        with open(file_path, "rb") as file:
            return file.read().decode("utf-8")

    if function_name == "Write":
        file_path = arguments["file_path"]
        content = arguments["content"]

        parent_directory = os.path.dirname(file_path)
        if parent_directory:
            os.makedirs(parent_directory, exist_ok=True)

        # newline="" prevents Python from changing newline characters.
        with open(file_path, "w", encoding="utf-8", newline="") as file:
            file.write(content)

        return f"Successfully wrote content to {file_path}"

    raise RuntimeError(f"Unknown tool: {function_name}")


def assistant_message_to_dict(message):
    result = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in (message.tool_calls or [])
        ],
    }

    if message.content is not None:
        result["content"] = message.content

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True)
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    messages = [
        {
            "role": "user",
            "content": args.p,
        }
    ]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        assistant_message = chat.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        messages.append(assistant_message_to_dict(assistant_message))

        if not tool_calls:
            if assistant_message.content is not None:
                sys.stdout.write(assistant_message.content)
            return

        for tool_call in tool_calls:
            result = execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    main()