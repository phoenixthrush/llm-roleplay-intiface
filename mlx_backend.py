import asyncio
import json
from pathlib import Path

from buttplug import ButtplugClient, DeviceOutputCommand, OutputType
from buttplug.errors import ButtplugDeviceError, ButtplugError
from mlx_lm import generate, load

# -----------------------------
# Config
# -----------------------------

MODEL_REPO = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
SERVER_URL = "ws://127.0.0.1:12345"
MODELFILE_PATH = Path("Modelfile")

ALLOWED_ACTIONS = {"vibrate", "rotate", "stop"}
ACTION_OUTPUTS = {
    "vibrate": OutputType.VIBRATE,
    "rotate": OutputType.ROTATE,
}


# -----------------------------
# Setup
# -----------------------------

model, tokenizer = load(MODEL_REPO)


# -----------------------------
# Prompting
# -----------------------------

JSON_RULES = """
Return exactly one JSON object with this shape:
{
  "reply": "short natural reply",
  "actions": [
    {
      "type": "vibrate | rotate | stop",
      "value": 0
    }
  ]
}

Rules:
- always include reply
- always include actions
- actions must be a JSON array
- use an empty array when no device control is needed
- use value as an integer from 0 to 100 for vibrate and rotate
- use value as null for stop
- only use vibrate, rotate, or stop
- do not put raw line breaks inside JSON strings
- do not escape apostrophes
- for normal conversation, actions should usually be an empty array
- do not explain the JSON
- do not add any text before or after the JSON
- do not use terminal formatting or control characters

Action meanings:
- vibrate: gentle or pulsing stimulation
- rotate: stronger or more active motion
- stop: stop all current device output
""".strip()


def load_system_prompt() -> str:
    """Load the system prompt."""
    base_prompt = ""

    if MODELFILE_PATH.exists():
        base_prompt = MODELFILE_PATH.read_text().strip()

    if base_prompt:
        return f"{base_prompt}\n\n{JSON_RULES}"

    return JSON_RULES


SYSTEM_PROMPT = load_system_prompt()


def build_messages(history: list, user_input: str) -> list:
    """Build messages for chat."""
    instruction = f"{SYSTEM_PROMPT}\n\nUser message:\n{user_input}\n\nReturn JSON only."

    if not history:
        return [{"role": "user", "content": instruction}]

    return [
        *history,
        {"role": "user", "content": instruction},
    ]


# -----------------------------
# JSON parsing
# -----------------------------


def extract_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def clean_string(text: str) -> str:
    """Remove unwanted control characters from a string."""
    cleaned = []

    for char in text:
        if char in "\n\r\t":
            cleaned.append(char)
            continue

        if ord(char) < 32:
            continue

        cleaned.append(char)

    return "".join(cleaned).strip()


def clean_data(value):
    """Clean strings inside nested data."""
    if isinstance(value, str):
        return clean_string(value)

    if isinstance(value, list):
        return [clean_data(item) for item in value]

    if isinstance(value, dict):
        return {key: clean_data(item) for key, item in value.items()}

    return value


def parse_response(content: str) -> dict:
    """Parse and clean a model response."""
    json_text = extract_json_object(content)
    if json_text is None:
        raise json.JSONDecodeError("Could not find JSON object", content, 0)

    data = json.loads(json_text)
    return clean_data(data)


# -----------------------------
# Validation
# -----------------------------


def validate_actions(actions: list) -> list:
    """Validate action data."""
    if not isinstance(actions, list):
        return []

    valid_actions = []

    for item in actions:
        if not isinstance(item, dict):
            continue

        action_type = item.get("type")
        value = item.get("value")

        if action_type not in ALLOWED_ACTIONS:
            continue

        if action_type == "stop":
            valid_actions.append({"type": "stop", "value": None})
            continue

        if not isinstance(value, int):
            continue

        valid_actions.append(
            {
                "type": action_type,
                "value": max(0, min(100, value)),
            }
        )

    return valid_actions


def validate_response(data: dict) -> dict:
    """Validate a model response."""
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        reply = "Okay."

    return {
        "reply": reply,
        "actions": validate_actions(data.get("actions")),
    }


# -----------------------------
# MLX chat
# -----------------------------


def ask_mlx(history: list, user_input: str) -> tuple[dict | None, str]:
    """Ask the MLX model for a structured response."""
    messages = build_messages(history, user_input)

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
    )

    raw_content = response.strip()

    try:
        parsed = parse_response(raw_content)
    except json.JSONDecodeError:
        return None, raw_content

    return validate_response(parsed), raw_content


# -----------------------------
# Buttplug
# -----------------------------


def print_device_capabilities(device) -> None:
    """Print device capabilities."""
    outputs = []

    if device.has_output(OutputType.VIBRATE):
        outputs.append("Vibrate")
    if device.has_output(OutputType.ROTATE):
        outputs.append("Rotate")

    print(f"  {device.name}")
    if outputs:
        print(f"    Outputs: {', '.join(outputs)}")
    print()


async def connect_client() -> ButtplugClient | None:
    """Connect to Intiface Central."""
    client = ButtplugClient("MLX Device Controller")

    client.on_device_added = lambda d: print(f"\n[+] Device connected: {d.name}")
    client.on_device_removed = lambda d: print(f"\n[-] Device disconnected: {d.name}")
    client.on_disconnect = lambda: print("\n[!] Server connection lost!")

    print("Connecting to Intiface Central...")

    try:
        await client.connect(SERVER_URL)
    except ButtplugError as e:
        print("ERROR: Could not connect to Intiface Central!")
        print(f"Error: {e}")
        return None

    print("Connected!\n")
    return client


async def scan_devices(client: ButtplugClient) -> list:
    """Scan for devices."""
    print("Scanning for devices...")
    print("Turn on your Bluetooth/USB devices now.\n")

    await client.start_scanning()
    await asyncio.to_thread(input, "Press Enter when your devices are connected...")
    await client.stop_scanning()

    devices = list(client.devices.values())

    if not devices:
        print("No devices found.")
        return []

    print(f"\nFound {len(devices)} device(s):\n")
    for device in devices:
        print_device_capabilities(device)

    return devices


async def run_actions(client: ButtplugClient, actions: list) -> None:
    """Run validated actions."""
    devices = list(client.devices.values())

    for action in actions:
        action_type = action["type"]
        value = action["value"]

        if action_type == "stop":
            await client.stop_all_devices()
            continue

        output_type = ACTION_OUTPUTS.get(action_type)
        if output_type is None or value is None:
            continue

        intensity = value / 100.0

        for device in devices:
            if not device.has_output(output_type):
                continue

            try:
                await device.run_output(DeviceOutputCommand(output_type, intensity))
            except ButtplugDeviceError as e:
                print(f"\ndevice error on {device.name}: {e}")


# -----------------------------
# Chat loop
# -----------------------------


def add_history(history: list, user_input: str, result: dict) -> None:
    """Add a chat turn to history."""
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": json.dumps(result)})


async def read_input(prompt: str) -> str:
    """Read input without blocking the event loop."""
    return await asyncio.to_thread(input, prompt)


async def chat_loop(client: ButtplugClient) -> None:
    """Run the main chat loop."""
    history = []

    print("chat is ready")
    print("type 'q' to quit\n")

    while True:
        user_input = (await read_input("you> ")).strip()

        if not user_input:
            continue

        if user_input.lower() == "q":
            break

        try:
            result, raw_content = await asyncio.to_thread(
                ask_mlx,
                history,
                user_input,
            )

            if result is None:
                print("bot> Sorry, I returned invalid JSON.")
                print("raw output>")
                print(raw_content)
                print()
                continue

            print(f"bot> {result['reply']}\n")

            if result["actions"]:
                await run_actions(client, result["actions"])

            add_history(history, user_input, result)

        except ButtplugError as e:
            print(f"bot> device error: {e}\n")
        except Exception as e:
            print(f"bot> error: {e}\n")


# -----------------------------
# Main
# -----------------------------


async def main() -> None:
    """Run the app."""
    client = await connect_client()
    if client is None:
        return

    try:
        devices = await scan_devices(client)
        if not devices:
            await client.disconnect()
            return

        await chat_loop(client)

    finally:
        print("stopping devices and disconnecting...")

        try:
            await client.stop_all_devices()
        except Exception:
            pass

        try:
            await client.disconnect()
        except Exception:
            pass

        print("goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
