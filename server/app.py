import threading
import socketserver


class ThreadedUDPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request[0].strip()
        current_thread = threading.current_thread()
        print("Thread: {} client: {}, wrote: {}".format(
            current_thread.name, self.client_address, data))
        Split = threading.Thread(target=ParseIncomingData, args=(data,))
        Split.start()


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


def publish_messages(data):
    """Publishes multiple messages to a Pub/Sub topic."""
    print('Published {} .'.format(data))


def ParseIncomingData(message):
    sender = threading.Thread(target=publish_messages, args=(message,))
    sender.start()


if __name__ == "__main__":
    HOST, PORT = "0.0.0.0", 6071
    try:
        serverUDP = ThreadedUDPServer((HOST, PORT), ThreadedUDPRequestHandler)
        server_thread_UDP = threading.Thread(target=serverUDP.serve_forever)
        server_thread_UDP.daemon = True
        server_thread_UDP.start()
        serverUDP.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        serverUDP.shutdown()
        serverUDP.server_close()
        exit()
