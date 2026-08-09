import requests
try:
    from config import RAPIDAPI_KEY, RAPIDAPI_HOST
except ImportError:
    RAPIDAPI_KEY = "188eb7343dmsh65bd09ec77fe485p1cb072jsnc619e559af13"
    RAPIDAPI_HOST = "the-racing-api1.p.rapidapi.com"

url = f"https://{RAPIDAPI_HOST}/v1/racecards/basic"
headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
response = requests.get(url, headers=headers, params={"day": "today"}).json()

def inspect_node(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ["runners", "horses", "selections"] and isinstance(v, list) and len(v) > 0:
                print("--- SAMPLE RUNNER KEYS & VALUES ---")
                print(v[0])
                exit()
            inspect_node(v)
    elif isinstance(node, list):
        for item in node:
            inspect_node(item)

inspect_node(response)
