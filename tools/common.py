import asyncio
from pathlib import Path

class UDPRequestHandler(asyncio.DatagramProtocol):
    def __init__(self):
        super().__init__()

    def connection_made(self, transport):
        self.transport = transport

async def start_server(handler, addr):
    print_debug_banner("STARTING SERVER")

    loop = asyncio.get_running_loop()
    transport = await loop.create_datagram_endpoint(
        lambda: handler,
        local_addr = addr
    )

    try:
        await asyncio.sleep(3600)
    finally:
        transport.close()

# handle SIGKILL
def exit_handler(p):
    try:
        p.kill()
    except:
        print("Couldn't kill")

def get_labels(dirname):
    holds = [sign.name for sign in Path(dirname, 'holds_data').iterdir()]
    nonholds = [sign.name for sign in Path(dirname, 'nonholds_data').iterdir()]
    return [None] + sorted(holds + nonholds)

def print_debug_banner(msg):
    print(f"\n================={msg}===============\n")