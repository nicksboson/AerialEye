import os
from huggingface_hub import HfApi

# Create an API instance using the token you provided
api = HfApi(token="YOUR_TOKEN_HERE")

repo_id = "prfct-suraj/aerialeye-backend"

# List of files/folders to upload
files_to_upload = [
    "Dockerfile"
]

print("Starting upload to Hugging Face...")

for item in files_to_upload:
    if os.path.isdir(item):
        print(f"Uploading folder: {item}...")
        api.upload_folder(
            folder_path=item,
            path_in_repo=item.rstrip('/'),
            repo_id=repo_id,
            repo_type="space"
        )
    elif os.path.isfile(item):
        print(f"Uploading file: {item}...")
        api.upload_file(
            path_or_fileobj=item,
            path_in_repo=item,
            repo_id=repo_id,
            repo_type="space"
        )
    else:
        print(f"Skipping {item}, not found.")

print("Upload complete!")
