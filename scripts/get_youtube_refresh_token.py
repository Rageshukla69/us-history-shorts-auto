"""
Helper script (optional if you don't already have your YOUTUBE_REFRESH_TOKEN):
Run locally or in Colab to exchange your client_secrets.json for a permanent YOUTUBE_REFRESH_TOKEN.
"""
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main():
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    creds = flow.run_local_server(port=8080, access_type="offline", prompt="consent")
    print("\n=== COPY THESE INTO YOUR GITHUB SECRETS ===")
    print(f"YOUTUBE_CLIENT_ID: {creds.client_id}")
    print(f"YOUTUBE_CLIENT_SECRET: {creds.client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN: {creds.refresh_token}")


if __name__ == "__main__":
    main()
