import socket


def get_free_port() -> int:
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getsockname()
    sock.close()

    return port


if __name__ == '__main__':
    print(get_free_port())
