import argparse
from openai import OpenAI

def main(api_key: str, base_url: str):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    models = client.models.list()

    # depending on SDK version, you may need models.data
    for model in models.data if hasattr(models, "data") else models:
        print(model.id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List models using OpenAI Python client.")
    parser.add_argument("--api_key", required=True, help="API key for authentication")
    parser.add_argument(
        "--base_url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="Base URL for the API"
    )

    args = parser.parse_args()
    main(api_key=args.api_key, base_url=args.base_url)
