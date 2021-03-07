import asyncio

class UDPRequestHandler(asyncio.DatagramProtocol):
    def __init__(self):
        super().__init__()

    def connection_made(self, transport):
        self.transport = transport

async def start_server(handler, addr):
    print()
    print("==================starting server===============")
    print()

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