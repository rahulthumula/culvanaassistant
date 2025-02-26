import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/chat"
    async with websockets.connect(uri) as websocket:
        # Send a test message
        test_message = {"type": "audio", "message": "Hello, how are you doing?"}
        await websocket.send(json.dumps(test_message))
        print(f"Sent message: {test_message}")

        # Receive the response
        response = await websocket.recv()
        print(f"Received response: {response}")

asyncio.run(test_websocket())