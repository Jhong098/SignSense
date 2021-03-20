import asyncio
from pathlib import Path
import random
import time

SERVER_RECV_PORT = 9999
CLIENT_RECV_PORT = 9998

class UDPRequestHandler(asyncio.DatagramProtocol):
    def __init__(self):
        super().__init__()
    
    def connection_made(self, transport):
        self.transport = transport


def start_server(handler, addr):
    print_debug_banner("STARTING SERVER")

    loop = asyncio.get_event_loop()
    transport = loop.create_datagram_endpoint(
        lambda: handler,
        local_addr = addr
    )

    loop.run_until_complete(transport)
    loop.run_forever()


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


def print_prediction_probs(out, labels):
    print("--> ", end='')
    for i, label in enumerate(out):
        print("{}:{:.2f}% | ".format(labels[i], label*100), end='')
    print("\n")
