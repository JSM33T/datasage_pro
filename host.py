import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = os.getcwd()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving '{DIRECTORY}' at http://{socketserver.socket.gethostbyname(socketserver.socket.gethostname())}:{PORT}")
    httpd.serve_forever()
