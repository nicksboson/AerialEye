import json
from PIL import Image

def test_heuristic():
    try:
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color = 'green')
        print("PIL is working perfectly.")
    except Exception as e:
        print("Error", e)

if __name__ == '__main__':
    test_heuristic()
