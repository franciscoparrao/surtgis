#!/usr/bin/env python3
"""HTTP server with no-cache headers, CORS, and STAC proxy for SurtGIS demo."""
import http.server
import os
import urllib.request
import urllib.error
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SurtGISHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Proxy endpoint: /proxy?url=https://...
        if self.path.startswith('/proxy?url='):
            self.handle_proxy()
            return
        super().do_GET()

    def do_POST(self):
        if self.path.startswith('/proxy?url='):
            self.handle_proxy_post()
            return
        self.send_error(405)

    def handle_proxy(self):
        target_url = self.path[len('/proxy?url='):]
        # Decode until stable (avoid double-encoding)
        prev = None
        while target_url != prev:
            prev = target_url
            target_url = urllib.parse.unquote(target_url)
        try:
            req = urllib.request.Request(target_url, headers={
                'User-Agent': 'SurtGIS-Demo/0.3.0',
                'Accept': 'application/json, application/geo+json, */*',
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                self.send_response(200)
                ct = resp.headers.get('Content-Type', 'application/json')
                self.send_header('Content-Type', ct)
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def handle_proxy_post(self):
        target_url = self.path[len('/proxy?url='):]
        prev = None
        while target_url != prev:
            prev = target_url
            target_url = urllib.parse.unquote(target_url)
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len > 0 else b''
        try:
            req = urllib.request.Request(target_url, data=body, headers={
                'User-Agent': 'SurtGIS-Demo/0.3.0',
                'Accept': 'application/json, application/geo+json, */*',
                'Content-Type': self.headers.get('Content-Type', 'application/json'),
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                self.send_response(200)
                ct = resp.headers.get('Content-Type', 'application/json')
                self.send_header('Content-Type', ct)
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        if '/proxy' in str(args[0]):
            super().log_message(format, *args)

if __name__ == '__main__':
    server = http.server.HTTPServer(('0.0.0.0', 9999), SurtGISHandler)
    print('SurtGIS demo on http://localhost:9999 (no-cache + STAC proxy)')
    server.serve_forever()
