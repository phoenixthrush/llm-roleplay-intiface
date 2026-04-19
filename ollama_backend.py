import asyncio
import json
from pathlib import Path

from buttplug import ButtplugClient, DeviceOutputCommand, OutputType
from buttplug.errors import ButtplugDeviceError, ButtplugError
from ollama import chat

# -----------------------------
# Config
# -----------------------------

MODEL = "windffooxxyang1130764/mesugaki:4b"  # this model is still retarded but answers more reliably
# MODEL = "nemotron-mini"
SERVER_URL = "ws://127.0.0.1:12345"
MODELFILE_PATH = Path("Modelfile")

# TODO: add constrict later
ALLOWED_ACTIONS = {"vibrate", "rotate", "stop"}
ACTION_OUTPUTS = {
    "vibrate": OutputType.VIBRATE,
    "rotate": OutputType.ROTATE,
}


# -----------------------------
# Prompting
# -----------------------------

JSON_RULES = """
Return exactly one JSON object with this shape:
{
  "reply": "short natural reply",
  "actions": [
    {
      "type": "vibrate | rotate | constrict | stop",
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

Actions:
- vibrate -> used for touching sensation (hand, thighs, kiss)
- rotate -> literally straightaway handjob or fucking
- constrict -> only used if getting a blowjob
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
    """Build messages for Ollama."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"{user_input}\n\nReturn JSON only."},
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


def escape_newlines_in_strings(text: str) -> str:
    """Escape raw newlines inside JSON strings."""
    result = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                result.append(char)
                escape = False
                continue

            if char == "\\":
                result.append(char)
                escape = True
                continue

            if char == '"':
                result.append(char)
                in_string = False
                continue

            if char == "\n":
                result.append("\\n")
                continue

            if char == "\r":
                result.append("\\r")
                continue

            result.append(char)
            continue

        result.append(char)

        if char == '"':
            in_string = True

    return "".join(result)


def repair_json_text(text: str) -> str:
    """Repair a few common JSON mistakes."""
    text = text.strip()
    text = text.replace("\\'", "'")
    text = escape_newlines_in_strings(text)
    return text


def try_parse_json(text: str) -> dict | None:
    """Try to parse JSON."""
    for candidate in (text, repair_json_text(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def parse_response(content: str) -> dict:
    """Parse a model response as JSON."""
    parsed = try_parse_json(content)
    if parsed is not None:
        return parsed

    extracted = extract_json_object(content)
    if extracted is not None:
        parsed = try_parse_json(extracted)
        if parsed is not None:
            return parsed

    raise json.JSONDecodeError("Could not parse JSON", content, 0)


# -----------------------------
# Validation
# -----------------------------

"""
def looks_like_control_request(text: str) -> bool:
    "Check if the user likely asked for device control."
    text = text.lower()

    keywords = {
        "vibrate",
        "rotate",
        "stop",
        "more",
        "less",
        "slower",
        "faster",
        "lower",
        "higher",
        "start",
    }

    return any(word in text for word in keywords)
"""


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


def ask_ollama(history: list, user_input: str) -> tuple[dict | None, str]:
    """Ask Ollama for a structured response."""
    response = chat(
        model=MODEL,
        messages=build_messages(history, user_input),
    )

    raw_content = response.message.content.strip()

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
    client = ButtplugClient("Ollama Device Controller")

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
                ask_ollama,
                history,
                user_input,
            )

            if result is None:
                print("alice> Sorry, I returned invalid JSON.")
                print("raw output>")
                print(raw_content)
                print()
                continue

            # if not looks_like_control_request(user_input):
            #    result["actions"] = []

            print(f"alice> {result['reply']}\n")

            if result["actions"]:
                await run_actions(client, result["actions"])

            add_history(history, user_input, result)

        except ButtplugError as e:
            print(f"alice> device error: {e}\n")
        except Exception as e:
            print(f"alice> error: {e}\n")


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
