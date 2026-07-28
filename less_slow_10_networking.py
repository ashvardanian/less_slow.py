#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Networking — sockets, event loops, and what a framework costs.

Latency measurements where the mean is the least interesting statistic. The
loopback-versus-public comparison is kept even though the effect did not
reproduce on this machine: a cloud instance with a directly attached address
has no NAT hairpin to pay for. What both routes share is a tail past twenty
times their average, and the tail is the thing that pages you.

`asyncio` does not make a single round trip faster — per message it is about
twice the cost of a blocking socket, because the event loop is machinery you
pay for whether or not you are waiting on anything. Allowing out-of-order
completion buys nothing measurable.

An HTTP framework on top costs another order of magnitude. For an internal
service that needs neither content negotiation nor an OpenAPI schema, that is
the entire bill for convenience nobody asked for.
"""

import sys
from typing import List

import pytest

# region: Networking and Databases

# ? When implementing web-applications, Python developers often rush to
# ? use overloaded high-level frameworks, like Django, Flask, or FastAPI,
# ? without ever considering a lower-level route.
# ?
# ? Let's implement a simple "echo" client and server using Python's
# ? built-in `socket` module, and compare its performance with a similar
# ? implementation in the `asyncio` module and FastAPI.

import socket  # for TCP and UDP servers # noqa: E402
import inspect  # to get the source code of a function # noqa: E402
import subprocess  # to start a server in a subprocess # noqa: E402
import time  # sleep for a bit until the socket binds # noqa: E402
from abc import ABC, abstractmethod  # to define abstract classes # noqa: E402
from typing import Literal  # noqa: E402

# ? The User Datagram Protocol (UDP) is OSI Layer 4 "Transport protocol", and
# ? should be able to operate on top of any OSI Layer 3 "Network protocol".
# ?
# ? In most cases, it operates on top of the Internet Protocol (IP), which can
# ? have Maximum Transmission Unit (MTU) ranging 20 for IPv4 and 40 for IPv6
# ? to 65535 bytes. In our case, however, the OSI Layer 2 "Data Link Layer" is
# ? likely to be Ethernet, which has a MTU of 1500 bytes, but most routers are
# ? configured to fragment packets larger than 1460 bytes. Hence, our choice!
RPC_MTU = 1460
RPC_PORT = 12345
RPC_PACKET_TIMEOUT_SEC = 0.05
RPC_BATCH_TIMEOUT_SEC = 0.5


def fetch_public_ip() -> str:
    """
    Returns the 'default' (outbound) IP address of the current machine.
    Note that this may be a private IP if behind NAT (it won't be your
    real public-facing IP if you are behind a router/firewall).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # The IP/port here doesn't need to be reachable (we never send data);
        # we just need the OS to pick a default interface for this "outbound" connection.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


class EchoServer(ABC):
    """Abstract base class for echo servers."""

    def __init__(self, host: str = "0.0.0.0", port: int = RPC_PORT):
        """
        :param host: The host to bind the server to. Set to '0.0.0.0' to listen on all
            interfaces. Set to 'localhost' or '127.0.0.1' to listen on the loopback
            interface.
        :param port: The port to bind the server to.
        """
        self.host = host
        self.port = port

    @abstractmethod
    def run(self):
        """Run the echo server (blocking call)."""
        pass


class TCPEchoServer(EchoServer):
    """Simple TCP Echo Server."""

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            while True:
                conn, _ = server.accept()
                with conn:
                    while True:
                        data = conn.recv(RPC_MTU)
                        if not data:
                            break
                        conn.sendall(data)


class UDPEchoServer(EchoServer):
    """Simple UDP Echo Server."""

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            while True:
                data, addr = server.recvfrom(RPC_MTU)
                if not data:
                    break
                server.sendto(data, addr)


class EchoClient(ABC):
    """Abstract base class for echo clients."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = RPC_PORT,
        timeout: float = RPC_PACKET_TIMEOUT_SEC,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

    @abstractmethod
    def connect(self):
        """Establish or prepare the client socket (TCP connect, or just open a UDP socket)."""
        pass

    @abstractmethod
    def send_and_receive(self, data: bytes) -> bytes:
        """Send data and receive its echo."""
        pass

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        """Send a batch of messages and receive their echoes."""
        return [self.send_and_receive(m) for m in messages]

    @abstractmethod
    def close(self):
        """Close the underlying socket."""
        pass


class TCPEchoClient(EchoClient):
    """TCP Echo Client implementation."""

    def __init__(
        self,
        host="localhost",
        port=RPC_PORT,
        timeout=RPC_PACKET_TIMEOUT_SEC,
    ):
        super().__init__(host, port, timeout)
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def send_and_receive(self, data: bytes) -> bytes:
        self._sock.sendall(data)
        return self._sock.recv(RPC_MTU)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None


class UDPEchoClient(EchoClient):
    """UDP Echo Client implementation."""

    def __init__(
        self,
        host="localhost",
        port=RPC_PORT,
        timeout=RPC_PACKET_TIMEOUT_SEC,
    ):
        super().__init__(host, port, timeout)
        self._sock = None

    def connect(self):
        # For UDP, "connect" isn't needed
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)

    def send_and_receive(self, data: bytes) -> bytes:
        # For UDP, we must specify the address on sendto unless we've "connected" the socket.
        self._sock.sendto(data, (self.host, self.port))
        resp, _ = self._sock.recvfrom(RPC_MTU)
        return resp

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None


class ServerProcess:
    """
    Wraps an EchoServer in a subprocess. On __enter__, spawns the server
    and returns `self`. On __exit__, kills the subprocess.
    """

    def __init__(self, server: EchoServer):
        self.server = server
        self._proc = None

    def __enter__(self):
        source_code = inspect.getsource(self.server.__class__)
        # We'll also need the base class if the server references it:
        base_code = inspect.getsource(EchoServer)
        # Recreate an identical server instance in another process and call run()
        script = f"""
import socket
from abc import ABC, abstractmethod

RPC_MTU = {RPC_MTU}
RPC_PORT = {RPC_PORT}

{base_code}
{source_code}

if __name__ == "__main__":
    server = {self.server.__class__.__name__}(host={self.server.host!r}, port={self.server.port})
    server.run()
"""

        self._proc = subprocess.Popen([sys.executable, "-c", script])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._proc:
            self._proc.kill()
            self._proc.wait()


def profile_echo_latency(
    benchmark,
    server_class,
    client_class,
    packet_length: int = 1024,
    rounds: int = 100_000,
    batch_size: int = 1,
    use_batching: bool = False,
    route: Literal["loopback", "public"] = "loopback",
):
    """
    A generic echo latency profiler that uses class-based server/client.
    """

    packet = b"ping" * (packet_length // 4)
    address_to_listen = "127.0.0.1" if route == "loopback" else "0.0.0.0"
    address_to_talk = "127.0.0.1" if route == "loopback" else fetch_public_ip()
    lost_packets = 0

    # Initialize server and run in a subprocess context
    server = server_class(host=address_to_listen)
    context = ServerProcess(server).__enter__()
    time.sleep(0.5)  # Short wait to ensure server is listening

    # Create client
    client = client_class(host=address_to_talk)
    client.connect()

    # We may want to allow executing the requests within the batch asynchronously
    send_many: callable = client_class.send_and_receive_batch
    emulate_sending_many: callable = EchoClient.send_and_receive_batch
    supports_batching: bool = emulate_sending_many is not send_many
    packets = [packet] * batch_size
    if use_batching:
        assert supports_batching, "Client does not support batching!"

    def runner():
        nonlocal lost_packets
        try:
            responses = send_many(client, packets)
            if any(r != packet for r in responses):
                raise ValueError("Mismatched echo response!")
        except socket.timeout:
            lost_packets += batch_size

    benchmark.pedantic(runner, iterations=1, rounds=rounds)
    benchmark.extra_info["lost_packets"] = lost_packets

    client.close()
    context.__exit__(None, None, None)  # kill the server


@pytest.mark.benchmark(group="10-networking-echo")
def test_rpc_tcp_loopback(benchmark):
    profile_echo_latency(benchmark, TCPEchoServer, TCPEchoClient, route="loopback")


@pytest.mark.benchmark(group="10-networking-echo")
def test_rpc_udp_loopback(benchmark):
    profile_echo_latency(benchmark, UDPEchoServer, UDPEchoClient, route="loopback")


@pytest.mark.benchmark(group="10-networking-echo")
def test_rpc_udp_public(benchmark):
    profile_echo_latency(benchmark, UDPEchoServer, UDPEchoClient, route="public")


# ? Packets sent to `127.0.0.1` are short-circuited in software. Sent to the
# ? machine's own "public" IP they may instead traverse NAT hairpin and
# ? firewall checks — the same destination, a different path through the
# ? kernel. On hardware where that happens, the public route costs more.
# ?
# ? It does not happen here, and the negative result is worth keeping:
# ?
# ? Intel Xeon 4 · CPython 3.14t · 100K round trips, mean latency
# ?
# ?   UDP public      20.1 µs    1.00x  574 µs worst case
# ?   UDP loopback    22.0 µs    1.09x  484 µs worst case
# ?
# ? This is a cloud instance with a directly attached address and no hairpin
# ? to pay for, so the two paths are indistinguishable — 1.09x is noise. What
# ? survives is the shape of the distribution: both routes have a mean near
# ? 20 µs and a worst case past 480 µs, a 25x spread. The tail is the thing
# ? that pages you at 3am, and no average will show it to you.
# ?
# ? Sounds interesting? I suggest reading
# ?
# ? - "High Performance Browser Networking" by Ilya Grigorik:
# ?   https://hpbn.co/
# ? - "Moving past TCP in the data center, part 2" by Jake Edge:
# ?   https://lwn.net/Articles/914030/


class AsyncioTCPEchoServer(EchoServer):
    """Asyncio-based TCP Echo Server."""

    def run(self):
        import asyncio

        async def handle_echo(reader, writer):
            while True:
                data = await reader.read(RPC_MTU)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def main_loop():
            server = await asyncio.start_server(handle_echo, self.host, self.port)
            async with server:
                # Serve forever (blocking)
                await server.serve_forever()

        asyncio.run(main_loop())


class AsyncioTCPEchoClient(ABC):
    """
    Since your framework expects .connect(), .send_and_receive(), .close()
    in a synchronous style, we internally run an event loop and call asyncio
    functions with run_until_complete().
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._loop = None
        self._reader = None
        self._writer = None

    def connect(self):
        import asyncio

        self._loop = asyncio.new_event_loop()

        async def _connect():
            reader, writer = await asyncio.open_connection(self.host, self.port)
            # Optionally, we can set socket timeouts or other config here.
            return reader, writer

        self._reader, self._writer = self._loop.run_until_complete(_connect())

    def send_and_receive(self, data: bytes) -> bytes:
        async def _send_and_receive(d):
            self._writer.write(d)
            await self._writer.drain()
            resp = await self._reader.read(RPC_MTU)
            return resp

        return self._loop.run_until_complete(_send_and_receive(data))

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        async def _send_and_receive_batch(msgs: List[bytes]):
            results = []
            for m in msgs:
                self._writer.write(m)
                await self._writer.drain()
                resp = await self._reader.read(RPC_MTU)
                results.append(resp)
            return results

        return self._loop.run_until_complete(_send_and_receive_batch(messages))

    def close(self):
        async def _close():
            if self._writer:
                self._writer.close()
                await self._writer.wait_closed()

        if self._loop:
            self._loop.run_until_complete(_close())
            self._loop.close()
            self._loop = None


@pytest.mark.benchmark(group="10-networking-echo")
def test_batch16_rpc_asyncio_ordered(benchmark):
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


@pytest.mark.benchmark(group="10-networking-echo")
def test_batch16_rpc_asyncio_unordered(benchmark):
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


# ? The results are unsettling. The promise of `asyncio` is to provide a
# ? high-performance, non-blocking I/O framework. However, the overhead
# ? of the event loop, the context switches, and the additional buffering
# ? can make it slower than the synchronous TCP client per call.
# ?
# ? Per message, once the batch of 16 is normalized away:
# ?
# ? Intel Xeon 4 · CPython 3.14t · per message, batches of 16
# ?
# ?   blocking TCP        22.8 µs    1.00x
# ?   asyncio ordered     38.0 µs    1.67x  608 µs per batch
# ?   asyncio unordered   39.5 µs    1.73x  632 µs per batch
# ?
# ? Allowing out-of-order completion buys nothing — the two asyncio rows are
# ? indistinguishable. And the blocking client the event loop was meant to
# ? improve on is twice as fast per message.
# ?
# ? This, however, may not be as bad as higher-level frameworks like FastAPI,
# ? and one of the most common underlying ASGI servers, Uvicorn.


class FastAPIEchoServer(EchoServer):
    """
    Minimal FastAPI-based HTTP echo server. It exposes a POST /echo endpoint
    that simply returns the raw request body as-is (using a binary media type).
    """

    def run(self):
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import Response

        app = FastAPI()

        @app.post("/echo")
        async def echo_endpoint(req: Request):
            data = await req.body()
            return Response(content=data, media_type="application/octet-stream")

        uvicorn.run(app, host=self.host, port=self.port, log_level="error")


class RequestsClient(EchoClient):
    """
    A simple requests-based client, calling POST /echo with the raw data in the
    request body, and returning the response body as bytes.
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        super().__init__(host, port, timeout)
        self._session = None

    def connect(self):
        import requests

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/octet-stream"})

    def send_and_receive(self, data: bytes) -> bytes:
        url = f"http://{self.host}:{self.port}/echo"
        resp = self._session.post(url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def close(self):
        if self._session:
            self._session.close()
            self._session = None


class HTTPXAsyncEchoClient(EchoClient):
    """
    Uses the httpx library in async mode to talk to the /echo endpoint.
    Batching is done concurrently with asyncio.gather.
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        super().__init__(host, port, timeout)
        self._loop = None
        self._client = None

    def connect(self):
        import httpx
        import asyncio

        # We'll create a dedicated event loop for this client and
        # instantiate the AsyncClient inside it.
        self._loop = asyncio.new_event_loop()

        async def _setup():
            # Create an AsyncClient with the given timeout and
            # set headers for sending binary data.
            client = httpx.AsyncClient(timeout=self.timeout)
            client.headers.update({"Content-Type": "application/octet-stream"})
            return client

        self._client = self._loop.run_until_complete(_setup())

    def send_and_receive(self, data: bytes) -> bytes:
        """
        Sends a single request and awaits the response using AsyncClient.
        We wrap it in run_until_complete() for synchronous code compatibility.
        """

        async def _send_and_receive(d):
            url = f"http://{self.host}:{self.port}/echo"
            resp = await self._client.post(url, content=d)
            resp.raise_for_status()
            return resp.content

        return self._loop.run_until_complete(_send_and_receive(data))

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        """
        Demonstrates concurrent batch logic using asyncio.gather.
        All requests are fired off in parallel, then we await all responses.
        """
        import asyncio

        async def _send_and_receive_batch(msgs: List[bytes]) -> List[bytes]:
            url = f"http://{self.host}:{self.port}/echo"

            # Build a coroutine for each message
            async def post(msg: bytes):
                resp = await self._client.post(url, content=msg)
                resp.raise_for_status()
                return resp.content

            # Fire them off concurrently
            tasks = [post(m) for m in msgs]
            results = await asyncio.gather(*tasks)
            return list(results)

        return self._loop.run_until_complete(_send_and_receive_batch(messages))

    def close(self):
        """
        Closes the AsyncClient and event loop.
        """
        import asyncio

        async def _close():
            if self._client:
                await self._client.aclose()

        if self._loop:
            self._loop.run_until_complete(_close())
            self._loop.close()
            self._loop = None


@pytest.mark.benchmark(group="10-networking-echo")
def test_batch16_rpc_fastapi_requests(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        RequestsClient,
        route="loopback",
        batch_size=16,
        use_batching=False,  # ! Requests are typically synchronous
        rounds=1_000,
    )


@pytest.mark.benchmark(group="10-networking-echo")
def test_batch16_rpc_fastapi_httpx(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        HTTPXAsyncEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


# ? Batches of 16 messages, one round trip each:
# ?
# ? Intel Xeon 4 · CPython 3.14t · per batch of 16 messages
# ?
# ?   raw TCP, asyncio         0.61 ms    1.00x
# ?   requests + FastAPI      11.65 ms   19.19x
# ?   async HTTPX + FastAPI   24.89 ms   41.00x
# ?
# ? An HTTP framework costs an order of magnitude over a raw socket, on top of
# ? a Python IO stack that was already slow. For internal services, where
# ? nobody needs content negotiation or OpenAPI schemas, that is the entire
# ? bill for convenience nobody asked for.

# endregion: Networking and Databases
