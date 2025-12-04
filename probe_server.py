import socket
import requests

domain = "tushare.xcsc.com"
try:
    ip = socket.gethostbyname(domain)
    print(f"Domain {domain} resolves to {ip}")
except Exception as e:
    print(f"Could not resolve {domain}: {e}")
    ip = None

if ip:
    ports = [7172, 7173, 80, 8080]
    for port in ports:
        url = f"http://{domain}:{port}"
        try:
            print(f"Testing {url} ...")
            resp = requests.get(url, timeout=2)
            print(f"  Status: {resp.status_code}")
        except Exception as e:
            print(f"  Failed: {e}")

    # Test API endpoint specifically on 7172
    api_url = f"http://{domain}:7172"
    try:
        print(f"Testing API POST to {api_url} ...")
        # Tushare API expects a POST with JSON
        resp = requests.post(api_url, json={"api_name": "daily", "token": "test", "params": {}}, timeout=2)
        print(f"  Status: {resp.status_code}")
        print(f"  Content start: {resp.text[:100]}")
    except Exception as e:
        print(f"  Failed: {e}")
