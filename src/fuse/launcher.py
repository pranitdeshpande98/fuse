"""Start the whole of Fuse with one command:

    uv run fuse

It runs the Telegram bot and the background worker as two separate processes, so a slow
task never blocks the chat. If either one stops, the other is stopped too, so you never
end up with half of Fuse running without noticing. Ctrl+C stops both.
"""

import subprocess
import sys
import time

PROCESSES = {
    "bot": "fuse.channels.telegram_bot",
    "worker": "fuse.tasks.worker",
}


def main() -> None:
    running = {
        name: subprocess.Popen([sys.executable, "-m", module]) for name, module in PROCESSES.items()
    }
    print("Fuse is running (bot + worker). Press Ctrl+C to stop.", flush=True)
    exit_code = 0
    try:
        while True:
            for name, process in running.items():
                code = process.poll()
                if code is not None:
                    print(f"The {name} stopped (exit code {code}); stopping the rest.", flush=True)
                    exit_code = code or 1
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C in the same console reaches the children too; give them time to finish
        # their current step, then make sure nothing is left running.
        for process in running.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
