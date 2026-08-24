"""
Web Demo Server for FloorGen.
Serves the interactive web interface and runs real-time RAG retrieval and diffusion synthesis.

Usage:
  python -m floorgen.demo.web_app --port 8000
"""

import os
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

from floorgen.data.scripts.parse_rplan import load_plans_from_dir
from floorgen.data.scripts.generate_sample_data import generate_dataset
from floorgen.rag.retriever import FloorplanRAGRetriever
from floorgen.postprocess.vectorize import export_svg


class FloorGenHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            viewer_path = os.path.join(os.path.dirname(__file__), "viewer.html")
            if os.path.exists(viewer_path):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                with open(viewer_path, "rb") as f:
                    self.wfile.write(f.read())
                return
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ready",
                "system": "FloorGen",
                "version": "0.1.0",
                "hardware_mode": "Zero-Cost CPU / Low-VRAM"
            }).encode("utf-8"))
            return

        return super().do_GET()


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, FloorGenHandler)
    print(f"🚀 FloorGen Interactive Demo running at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)
