#!/usr/bin/env python3
import argparse
import os
import zipfile
import tempfile
import requests
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import getpass


class Publisher:
    def __init__(self):
        self.ignore_patterns = [
            ".venv",
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".DS_Store",
            "node_modules/",
            ".env",
            "teams.json",
            "submissions/",
            "leaderboard/",
            "tests",
            "*.zip",
        ]

    def should_ignore(self, file_path):
        """Check if file should be ignored based on ignore patterns"""
        for pattern in self.ignore_patterns:
            if pattern.endswith("/"):
                # Directory pattern
                if pattern[:-1] in file_path.parts:
                    return True
            elif "*" in pattern:
                # Wildcard pattern
                if pattern.startswith("*"):
                    if file_path.name.endswith(pattern[1:]):
                        return True
            else:
                # Exact match
                if pattern in str(file_path):
                    return True
        return False

    def create_zip(self, source_dir, zip_path):
        """Create a zip file of the repository"""
        print(f"Creating zip file from {source_dir}...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zipf:
            source_path = Path(source_dir)

            for file_path in source_path.rglob("*"):
                if file_path.is_file() and not self.should_ignore(
                    file_path.relative_to(source_path)
                ):
                    # Add file to zip with relative path
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

        print(f"Zip file created: {zip_path}")

    def encrypt_file(self, file_path, key):
        """Encrypt a file using the pre-shared key"""
        print("Encrypting submission...")

        # Convert key to Fernet key
        fernet_key = base64.urlsafe_b64encode(
            base64.urlsafe_b64decode(key.encode())[:32]
        )
        fernet = Fernet(fernet_key)

        # Read and encrypt file
        with open(file_path, "rb") as f:
            file_data = f.read()

        encrypted_data = fernet.encrypt(file_data)

        # Write encrypted file
        encrypted_path = file_path + ".encrypted"
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)

        print("Encryption completed.")
        return encrypted_path

    def send_submission(self, server, port, team_name, encrypted_file_path):
        """Send encrypted submission to server"""
        url = f"https://{server}:{port}/submit/"

        print(f"Sending submission to {url}...")

        try:
            with open(encrypted_file_path, "rb") as f:
                files = {"submission": f}
                data = {"team_name": team_name}

                response = requests.post(url, files=files, data=data, timeout=6000)

            if response.status_code == 200:
                print("✅ Submission sent successfully!")
                print(f"Server response: {response.json().get('message', 'Success')}")
                return True
            else:
                print(f"❌ Submission failed: {response.status_code}")
                try:
                    error_msg = response.json().get("error", "Unknown error")
                    print(f"Error: {error_msg}")
                except:
                    print(f"Error: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return False

    def publish(self, server, port, source_dir=None, team_name=None, key=None):
        """Main publish function"""
        if source_dir is None:
            source_dir = os.getcwd()

        # Get team name and key from parameters or user input
        if not team_name:
            team_name = input("Enter team name: ").strip()
        if not team_name:
            print("Error: Team name is required")
            return False

        if not key:
            key = getpass.getpass("Enter pre-shared key: ").strip()
        if not key:
            print("Error: Pre-shared key is required")
            return False

        # Create temporary directory for zip and encrypted files
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "submission.zip")

            try:
                # Create zip file
                self.create_zip(source_dir, zip_path)

                # Encrypt zip file
                encrypted_path = self.encrypt_file(zip_path, key)

                # Send to server
                success = self.send_submission(server, port, team_name, encrypted_path)

                return success

            except Exception as e:
                print(f"❌ Error during publish: {e}")
                return False


def main():
    parser = argparse.ArgumentParser(description="Publish submission to server")
    parser.add_argument(
        "--server",
        default="haih25.events.semron.org",
        help="IP address of server running submission_receiver.py",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=443,
        help="Port at server at which submission_receiver.py runs",
    )
    parser.add_argument(
        "--source", help="Source directory to zip (default: current directory)"
    )
    parser.add_argument("--team-name", help="Team name (will prompt if not provided)")
    parser.add_argument("--key", help="Pre-shared key (will prompt if not provided)")

    args = parser.parse_args()

    publisher = Publisher()
    success = publisher.publish(
        args.server, args.port, args.source, args.team_name, args.key
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
