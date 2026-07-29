#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Networking — sockets, event loops, and what a framework costs.

Latency measurements where the mean is the least interesting statistic. Every
route here is flat to p99 and then rises by an order of magnitude: p99.9 is
roughly triple the median and the worst case is 12-28x it. A request that
issues ten calls meets its p99.9 almost every time, so an average latency
figure describes a case that rarely happens.

The loopback-versus-public comparison is a documented non-reproduction — this
cloud instance has a directly attached address and no NAT hairpin — and TCP
against UDP is another, at these sizes. Both are worth keeping, because a
transport chosen for speed it does not deliver still costs the ordering and
delivery guarantees given up for it.

`asyncio` does not make a round trip faster. It makes several possible at
once, and the two are easy to confuse: sixteen messages sent in lockstep cost
53.1 µs each, the same sixteen written before any reply is read cost 11.8 µs
each, and a blocking socket sits between them at 23.6 µs. Whether an event
loop helps or hurts depends entirely on whether anything overlaps.

The same confusion appears one layer up, and costs more. Sixteen concurrent
`httpx` requests against a single-worker uvicorn are 1.95x slower than sixteen
blocking ones, because the client is concurrent and the server is not. An HTTP
framework on top of that is 58-112x a raw socket.
"""

import sys
from typing import List

import pytest

# region: Networking

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

# ? Payload size is chosen by the narrowest link on the path, not by the
# ? protocol. Ethernet carries 1500 bytes of payload, and every header above it
# ? eats into that:
# ?
# ?   Ethernet MTU              1500 B
# ?   − IPv4 header               20 B  → 1480 for the transport layer
# ?   − TCP header                20 B  → 1460 usable   ← the TCP MSS
# ?   − UDP header                 8 B  → 1472 usable
# ?
# ? Exceed the link MTU and a router fragments, which turns one lost packet
# ? into a lost message. IPv6 refuses to fragment in transit at all and
# ? requires a minimum MTU of 1280, so the safe payload shrinks further the
# ? moment the path stops being local.
RPC_MTU = 1460  # the TCP MSS on Ethernet, used here as a receive-buffer size
RPC_PORT = 12345
RPC_PACKET_TIMEOUT_SEC = 0.05


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

    transport = "tcp"

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


# ? An echo server is the smallest thing that measures a network: it does no
# ? work, so whatever the clock reports is the path itself. Two of them here,
# ? differing in what the kernel promises:
# ?
# ?   TCP   connection state, ordering, retransmission, flow control
# ?   UDP   a datagram arrives, or it does not
# ?
# ? TCP's guarantees are not free — a handshake to start, sequence numbers per
# ? segment, an acknowledgement path — and the usual conclusion is that UDP is
# ? the fast one. Whether that survives measurement at 1 KB on loopback is the
# ? question, and it is worth answering before giving up delivery guarantees.


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

    transport = "udp"

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            while True:
                data, addr = server.recvfrom(RPC_MTU)
                if not data:
                    break
                server.sendto(data, addr)


# ? The clients hide a real asymmetry behind one interface. `send_and_receive`
# ? means something different on each transport:
# ?
# ?   TCP   sendall, then recv — the reply is a stream, and one recv may
# ?         return part of a message, or two messages stuck together
# ?   UDP   sendto, then recvfrom — one datagram in, one out, or nothing
# ?
# ? At 1 KB on loopback a single `recv` happens to return exactly one message
# ? every time, which is why this code works and why code shaped like it fails
# ? in production. The `asyncio` client further down uses `readexactly` and is
# ? the only one here that is actually correct about framing.


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


# ? A server has to run somewhere other than the process measuring it, or the
# ? benchmark would be timing its own event loop. This one is rebuilt from
# ? source in a fresh interpreter:
# ?
# ?   parent                          child
# ?   ──────                          ─────
# ?   inspect.getsource(EchoServer)   exec'd as a script
# ?   inspect.getsource(subclass)     socket, ABC, RPC_MTU, RPC_PORT
# ?                                   ↓
# ?   client ───→ 127.0.0.1:12345 ──→ server.run()
# ?
# ? Only four names cross that line. A server class that references anything
# ? else at module level — a helper, a constant, an import at the top of this
# ? module — raises `NameError` in a subprocess whose stderr goes nowhere, and
# ? surfaces to the reader as a connection timeout half a minute later.
# ?
# ? That is why every server below imports what it needs inside `run`.


class ServerProcess:
    """
    Wraps an EchoServer in a subprocess. On __enter__, spawns the server
    and returns `self`. On __exit__, kills the subprocess.

    Servers are rebuilt from source, so they may reference only `socket`,
    `ABC`, `abstractmethod`, `RPC_MTU` and `RPC_PORT` at module level. Anything
    else has to be imported inside `run`.
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


def _await_server(
    host: str,
    port: int = RPC_PORT,
    transport: str = "tcp",
    timeout: float = 15.0,
) -> None:
    """Poll until the server answers, instead of guessing at a sleep.

    A fixed `time.sleep` is a race in both directions: too long for a socket
    server that binds in a millisecond, and too short for the FastAPI one,
    whose subprocess imports uvicorn and pydantic before it listens.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            if transport == "udp":
                # ! UDP has nothing to connect to, so the probe is a real
                # ! datagram — the echo coming back is the readiness signal.
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.settimeout(0.1)
                    probe.sendto(b"ready?", (host, port))
                    probe.recvfrom(RPC_MTU)
            else:
                with socket.create_connection((host, port), timeout=0.2):
                    pass
            return
        except OSError:
            time.sleep(0.005)
    raise TimeoutError(f"{transport} server on {host}:{port} never answered")


# ? Latency is the one thing in this suite that cannot be batched. Every other
# ? chapter runs its kernel thousands of times and divides, because the
# ? per-call clock resolution would otherwise dominate — but averaging round
# ? trips destroys exactly the distribution worth seeing.
# ?
# ?   benchmark(kernel)                 many calls per sample → a mean
# ?   pedantic(iterations=1, rounds=N)  one call per sample   → N samples
# ?
# ? So this uses `pedantic` with a single iteration per round, and keeps every
# ? sample. A round trip is ~20 µs against a ~140 ns harness cost, so there is
# ? no accuracy to gain from batching and a tail to lose.


def profile_echo_latency(
    benchmark,
    server_class,
    client_class,
    packet_length: int = 1024,
    rounds: int = 100_000,
    batch_size: int = 1,
    batch_method: str = "send_and_receive_batch",
    route: Literal["loopback", "public"] = "loopback",
):
    """
    A generic echo latency profiler that uses class-based server/client.
    """

    packet = b"ping" * (packet_length // 4)
    address_to_listen = "127.0.0.1" if route == "loopback" else "0.0.0.0"
    address_to_talk = "127.0.0.1" if route == "loopback" else fetch_public_ip()
    lost_packets = 0
    latencies: List[float] = []

    send_many = getattr(client_class, batch_method)
    packets = [packet] * batch_size
    server = server_class(host=address_to_listen)

    # ! `with`, not a bare `__enter__`. Without it, any exception below — a
    # ! mismatched echo, a failed assert, an interrupt — leaks a subprocess
    # ! still holding the port, and every later test in this module then dies
    # ! with EADDRINUSE. One failure would become six.
    with ServerProcess(server) as context:
        _await_server(address_to_talk, transport=server.transport)
        client = client_class(host=address_to_talk)
        client.connect()
        try:

            def runner():
                nonlocal lost_packets
                started = time.perf_counter()
                try:
                    responses = send_many(client, packets)
                    if any(response != packet for response in responses):
                        raise ValueError("Mismatched echo response!")
                except socket.timeout:
                    lost_packets += batch_size
                latencies.append((time.perf_counter() - started) * 1e6)

            benchmark.pedantic(runner, iterations=1, rounds=rounds)
        finally:
            client.close()

    # ! The mean is the least interesting number a latency benchmark produces,
    # ! and it is the only one pytest-benchmark puts in `extra_info` by
    # ! default. Record the shape of the distribution instead.
    ordered = sorted(latencies)
    for label, quantile in (("p50", 0.50), ("p99", 0.99), ("p999", 0.999)):
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        benchmark.extra_info[f"{label}_us"] = round(ordered[index], 1)
    benchmark.extra_info["max_us"] = round(ordered[-1], 1)
    benchmark.extra_info["tail_ratio"] = round(
        ordered[-1] / ordered[len(ordered) // 2], 1
    )
    benchmark.extra_info["lost_packets"] = lost_packets


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
# ? Intel Xeon 4 • CPython 3.14t • 100K round trips, one 1 KB message each
# ?
# ?                      p50       p99     p99.9       max     max/p50
# ?   UDP public      19.9 µs   29.4 µs   62.7 µs   243 µs       12x
# ?   UDP loopback    22.5 µs   30.7 µs   63.5 µs   630 µs       28x
# ?   TCP loopback    23.2 µs   31.1 µs   60.3 µs   550 µs       24x
# ?
# ? This is a cloud instance with a directly attached address and no hairpin to
# ? pay for, so the routes are indistinguishable. TCP and UDP are too, which is
# ? its own answer to "should this be UDP" — the transport is not the cost at
# ? this size, and UDP buys the ordering and delivery problems for nothing.


# ? Read that table across rather than down. Every row is flat to p99 and then
# ? climbs by an order of magnitude — p99.9 is roughly triple the median, and
# ? the worst case is 12 to 28 times it.
# ?
# ? That shape decides how systems behave. A request that issues ten calls
# ? meets its p99.9 more often than not, so the tail of a dependency becomes
# ? the median of whatever depends on it. An average latency figure describes a
# ? case that rarely happens and hides the case that pages you.
# ?
# ? These percentiles come from timing each round trip and sorting them, not
# ? from the harness summary: a mean and a standard deviation cannot
# ? reconstruct a tail 28x the median, and reporting them implies a normal
# ? distribution that latency never has.
# ?
# ? Further reading:
# ?
# ? - "High Performance Browser Networking" by Ilya Grigorik:
# ?   https://hpbn.co/
# ? - "Moving past TCP in the data center, part 2" by Jake Edge:
# ?   https://lwn.net/Articles/914030/


# ? The same echo, through an event loop on both ends. Nothing about the
# ? protocol changes; what changes is who is waiting and whether they could be
# ? doing something else:
# ?
# ?   blocking   the thread stops at recv until bytes arrive
# ?   asyncio    the coroutine yields at await, the loop runs another
# ?
# ? With one connection and one outstanding message there is never another
# ? coroutine to run, so the loop is pure overhead. The interesting case is a
# ? batch, where sixteen messages could be in flight and the client gets to
# ? choose whether they are.


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


class AsyncioTCPEchoClient(EchoClient):
    """An `asyncio` client behind the same blocking interface as the others.

    Subclasses `EchoClient` rather than `ABC` directly: the profiler compares
    bound methods against the base class to decide what a client supports, and
    a class outside the hierarchy can never be classified correctly.
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

    # ? The next two methods send the same sixteen messages over the same
    # ? socket and differ only in ordering:
    # ?
    # ?   sequential   w r w r w r w r …    the wire is idle between each pair
    # ?   pipelined    w w w w … r r r r    the wire is busy until the reads
    # ?
    # ? Sequential is what `await` inside a loop produces, and it is the shape
    # ? most async code accidentally has — every `await` in a `for` body is a
    # ? full stall that the loop cannot fill, because there is nothing else
    # ? scheduled to run.

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        """Lockstep: write one, wait for its reply, then write the next."""

        async def _sequential(batch: List[bytes]):
            results = []
            for message in batch:
                self._writer.write(message)
                await self._writer.drain()
                results.append(await self._reader.readexactly(len(message)))
            return results

        return self._loop.run_until_complete(_sequential(messages))

    def send_and_receive_pipelined(self, messages: List[bytes]) -> List[bytes]:
        """Write the whole batch, then collect the replies."""

        async def _pipelined(batch: List[bytes]):
            for message in batch:
                self._writer.write(message)
            await self._writer.drain()
            # ! TCP is a stream, so sixteen echoed messages may arrive in any
            # ! number of reads. `readexactly` on the total is the only correct
            # ! way to reassemble them — a single `read(RPC_MTU)` would return
            # ! whatever happened to have arrived and silently desynchronize
            # ! the connection for every later round.
            expected = sum(len(message) for message in batch)
            blob = await self._reader.readexactly(expected)
            results, offset = [], 0
            for message in batch:
                results.append(blob[offset : offset + len(message)])
                offset += len(message)
            return results

        return self._loop.run_until_complete(_pipelined(messages))

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
def test_batch16_rpc_asyncio_sequential(benchmark):
    """Sixteen messages, each waiting for its own reply before the next goes out."""
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        batch_method="send_and_receive_batch",
        rounds=1_000,
    )


@pytest.mark.benchmark(group="10-networking-echo")
def test_batch16_rpc_asyncio_pipelined(benchmark):
    """The same sixteen, all written before any reply is read."""
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        batch_method="send_and_receive_pipelined",
        rounds=1_000,
    )


# ? Intel Xeon 4 • CPython 3.14t • 16 messages of 1 KB, per batch and per message
# ?
# ?                        per batch    per message
# ?   asyncio pipelined       188 µs        11.8 µs    1.00x
# ?   blocking TCP                —         23.6 µs    2.01x
# ?   asyncio sequential      850 µs        53.1 µs    4.52x
# ?
# ? The two asyncio rows run identical code over identical sockets. The only
# ? difference is when the writes happen:
# ?
# ?   sequential   write → wait → write → wait → …    16 stalls
# ?   pipelined    write ×16 → drain → read all        1 stall
# ?
# ? Waiting for each echo before sending the next leaves the connection idle
# ? for a full round trip, sixteen times over. Writing everything first fills
# ? that idle time, and it is worth 4.52x.


# ? Note where that leaves the blocking client, which the event loop was
# ? supposed to improve on: pipelined `asyncio` is 2.01x faster per message,
# ? and sequential `asyncio` is 2.25x slower. Same library, opposite verdicts.
# ?
# ? So "is asyncio faster" has no answer. An event loop does not make a round
# ? trip quicker — it makes several possible at once, and a program that never
# ? overlaps anything has bought the machinery and none of the benefit.


# ? Everything so far moved bytes. HTTP moves bytes plus a description of what
# ? they are, and the description is most of the cost:
# ?
# ?   raw socket   1024 bytes
# ?   HTTP POST    request line + headers + body + response line + headers
# ?                ↓
# ?                parse the request line, parse each header, match a route,
# ?                negotiate content type, validate, serialize, frame a reply
# ?
# ? All of that is worth paying for at a public boundary, where the caller is
# ? a browser, or someone else's client, or a version of your own service from
# ? two releases ago. Between two processes you control, it buys nothing that
# ? was in question.


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


# ? Two HTTP clients, and the difference between them is the point. `requests`
# ? issues sixteen requests one after another over a pooled connection. `httpx`
# ? issues all sixteen through `asyncio.gather`.
# ?
# ? The second should win. Whether it does depends on something outside the
# ? client entirely — how many of them the server can work on at once — and
# ? the server here is one uvicorn worker.


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
        rounds=1_000,
    )


# ? Intel Xeon 4 • CPython 3.14t • 16 messages of 1 KB, per batch
# ?
# ?   raw TCP, pipelined       0.19 ms     1.00x
# ?   requests + FastAPI      10.86 ms    57.74x
# ?   async HTTPX + FastAPI   21.15 ms   112.43x
# ?
# ? Two orders of magnitude for the same bytes. What the framework adds is
# ? request framing, header parsing, routing, content negotiation, validation
# ? and JSON — none of which an internal service between two of your own
# ? processes has any use for.
# ?
# ? The last row is the interesting one, because it is backwards. `httpx` with
# ? `asyncio.gather` is 1.95x *slower* than blocking `requests`, and the reason
# ? is that the server is a single uvicorn worker:
# ?
# ?   client concurrent   16 requests dispatched at once
# ?   server serial       1 worker, handling them one at a time
# ?
# ? Concurrency on one side of a connection is worth nothing if the other side
# ? is serial. `gather` adds sixteen tasks, sixteen connection checkouts and
# ? the scheduling to interleave them, and then they queue anyway. This is the
# ? most common way async makes a program slower — the code is concurrent, the
# ? bottleneck is not, and the coordination is pure overhead.

# endregion: Networking
