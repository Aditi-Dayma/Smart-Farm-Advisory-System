"""
Vercel Serverless Entry Point for Smart Farm Advisory System.

Satisfies Vercel's Python builder while serving the interactive agricultural
dashboard using Python's standard library (zero external dependencies required).
"""

from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"
if not HTML_PATH.exists():
    HTML_PATH = BASE_DIR / "frontend" / "index.html"


def get_html_bytes() -> bytes:
    if HTML_PATH.exists():
        return HTML_PATH.read_bytes()
    return b"<h1>Smart Farm Advisory System</h1><p>Dashboard loading...</p>"


# 1. WSGI application entrypoint (used by Vercel WSGI / Gunicorn / Flask-like runtimes)
def app(environ, start_response):
    status = "200 OK"
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("X-Content-Type-Options", "nosniff"),
    ]
    start_response(status, headers)
    return [get_html_bytes()]


# 2. BaseHTTPRequestHandler entrypoint (used by Vercel Serverless Functions)
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(get_html_bytes())

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()


# Alias for frameworks checking for application
application = app

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    port = 8000
    print(f"Serving Smart Farm Advisory System on http://localhost:{port}")
    with make_server("", port, app) as httpd:
        httpd.serve_forever()
