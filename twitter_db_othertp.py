from dotenv import load_dotenv
import requests
import tweepy
import datetime
import json
import os
import time
import asyncio
from postmarker.core import PostmarkClient
from openai import OpenAI
from prisma import Prisma
import logging

load_dotenv(".env", override=True)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Postmark and OpenAI clients
postmark = PostmarkClient(server_token=os.getenv("POSTMARK_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("POSTMARK_API_KEY"))

# Initialize Twitter client and Prisma ORM
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
prisma = Prisma()

cache_file = "tweets_cache.json"
cache_duration =  360  # 6 minutes


# Function to fetch unique Twitter handles and email subscriptions from DB using Prisma ORM
async def fetch_subscriptions_from_db():
    logger.info("Connecting to the database...")
    await prisma.connect()
    try:
        logger.info("Fetching subscriptions from the database...")
        subscriptions = await prisma.usertwittersubscription.find_many()
        logger.debug(f"Fetched subscriptions: {subscriptions}")

        subscription_data = []
        for sub in subscriptions:
            logger.info(f"Processing subscription for twitter: {sub}")
            subscription_data.append(
                {"twitter_users": sub.twitter_users, "email": sub.email}
            )

        logger.info(f"Returning subscription data: {subscription_data}")
        return subscription_data
    except Exception as e:
        logger.error(f"Error fetching subscriptions from DB: {e}")
    finally:
        await prisma.disconnect()
        logger.info("Disconnected from the database")


def fetch_coin_info(fetched_coin):
    try:
        # Fetch the data from DexScreener API
        response = requests.get(
            f"https://api.dexscreener.com/latest/dex/search?q={fetched_coin}"
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("pairs"):
            raise ValueError(f"No pairs found for the coin: {fetched_coin}")

        top_pair = data["pairs"][0]
        print(top_pair)  # Debug statement to inspect top pair

        # Get the base token symbol
        base_token_symbol = top_pair["baseToken"]["symbol"]

        # Define a dynamic default image URL based on the token symbol
        if base_token_symbol == "BTC":
            default_image_url = "https://www.blockchain.com/explorer/_next/static/media/btc.a6006067.png"
        elif base_token_symbol == "ETH":
            default_image_url = "https://dd.dexscreener.com/ds-data/chains/ethereum.png"
        elif base_token_symbol == "SOL":
            default_image_url = "https://dd.dexscreener.com/ds-data/chains/solana.png"
        else:
            default_image_url = "https://dd.dexscreener.com/ds-data/chains/default.png"

        # Extract coin image URL and fix if necessary
        image_url = top_pair.get("info", {}).get("imageUrl", default_image_url)

        # Check if the image URL starts with the wrong base URL and fix it
        if image_url.startswith("http://10.128.13.101:3000"):
            image_url = image_url.replace(
                "http://10.128.13.101:3000", "https://dd.dexscreener.com"
            )

        # Extract coin information with the corrected image URL
        coin_info = {
            "base_token_symbol": base_token_symbol,
            "price_usd": top_pair.get("priceUsd", None),
            "price_change_h24": top_pair.get("priceChange", {}).get("h24", None),
            "image_url": image_url,  # Use corrected or default image
            "website_url": next(
                (
                    site["url"]
                    for site in top_pair.get("info", {}).get("websites", [])
                    if site["label"] == "Website"
                ),
                None,
            ),
            "twitter_url": next(
                (
                    social["url"]
                    for social in top_pair.get("info", {}).get("socials", [])
                    if social["type"] == "twitter"
                ),
                None,
            ),
            "telegram_url": next(
                (
                    social["url"]
                    for social in top_pair.get("info", {}).get("socials", [])
                    if social["type"] == "telegram"
                ),
                None,
            ),
            "market_cap": top_pair.get("marketCap", None),
        }

        return coin_info

    except Exception as e:
        print(f"Error fetching coin info: {e}")
        return None

def generate_buy_sell_link(base_token_symbol):
    return f"https://beta.termix.ai/memecoin?data={{%22isTokenName%22:true,%22value%22:%22{base_token_symbol}%22,%22amount%22:%22%22}}"


def fetch_tweet_embed(tweet_url):
    try:
        response = requests.get(f"https://publish.twitter.com/oembed?url={tweet_url}")
        response.raise_for_status()
        return response.json().get("html")
    except Exception as e:
        print(f"Error fetching tweet embed: {e}")
        return None


# Function to infer the coin using OpenAI
def infer_coin(tweet_text):
    logger.info(f"Inferring coin from tweet: {tweet_text}")
    prompt = f"""
    Analyze the following tweet for any potential cryptocurrency names, acronyms, or phrases that might refer to tokens.
    Tweet: '{tweet_text}'

    Only Respond in JSON format:
    {{
        "coin_name": "<coin_name or 'None'>",
        "fetched_coin_symbol": "<fetched_coin_symbol or 'None'>"
    }}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the content safely
        content = response.choices[0].message.content.strip()

        # If the response is wrapped in a code block, strip it off
        if content.startswith("```json") and content.endswith("```"):
            content = content[8:-3].strip()  # Remove the ```json and ```

        coin_data = json.loads(content)
        return coin_data if coin_data["coin_name"] != "None" else None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e} - Content received: {content}")
        return None
    except Exception as e:
        logger.error(f"Error inferring coin: {e}")
        return None


# Function to fetch and process tweets
async def fetch_and_process_tweets():
    logger.info("Fetching and processing tweets...")
    subscriptions = await fetch_subscriptions_from_db()

    if not subscriptions:
        logger.error("No Twitter handles found in the database")
        return {"Error": "No Twitter handles found in the database"}

    twitter_handles_emails = {}
    for sub in subscriptions:
        for handle in sub["twitter_users"]:
            twitter_handles_emails[handle] = sub["email"]

    # Set the correct time window (last 30 minutes) for fetching tweets
    end_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=10
    )
    start_time = end_time - datetime.timedelta(minutes=5)

    tweets_data = []
    for handle in twitter_handles_emails:
        query = f"from:{handle} -is:retweet -is:quote -is:reply"
        logger.info(f"Fetching tweets for handle: {handle}")

        try:
            tweets = tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=["created_at", "text"],
                start_time=start_time,
                end_time=end_time,
                max_results=100,
            ).flatten(limit=500)

            for tweet in tweets:
                tweet_url = f"https://twitter.com/{handle}/status/{tweet.id}"
                logger.debug(f"Fetched tweet: {tweet.text} from {handle}")
                tweets_data.append(
                    {
                        "celebrity": handle,
                        "text": tweet.text,
                        "created_at": tweet.created_at.isoformat(),
                        "url": tweet_url,  # Add tweet URL for easier debugging
                    }
                )
        except Exception as e:
            logger.error(f"Error fetching tweets for {handle}: {e}")

    if not tweets_data:
        logger.info("No tweets fetched during this cycle.")

    logger.info(f"tweets_data: {tweets_data}")
    with open("tweets_data.json", "a") as f:
        json.dump(tweets_data, f)
        f.write(", \n")

    coins_data = []
    for tweet in tweets_data:
        coin_data = infer_coin(tweet["text"])
        if coin_data:
            coins_data.append(
                {
                    "twitter_handle": tweet["celebrity"],
                    "tweet": tweet["text"],
                    "fetched_coin": coin_data["coin_name"],
                    "fetched_coin_symbol": coin_data["fetched_coin_symbol"],
                    "email": twitter_handles_emails[tweet["celebrity"]],
                    "url": tweet["url"],
                }
            )
            logger.info(
                f"Coin found for tweet: {tweet['text']} by {tweet['celebrity']}: {coin_data['coin_name']}"
            )

            # Save coin data to a file for debugging purposes
            with open("coin_data_debug.json", "a") as f:
                json.dump(coins_data, f, default=str)
                f.write("\n")

    if not coins_data:
        logger.info("No coins found during this cycle.")

    return coins_data


def process_coins_data(coins_data):
    logger.info("Processing coins data...")
    processed_coin_data = []
    for data in coins_data:
        coin_symbol = data["fetched_coin_symbol"]
        coin_info = fetch_coin_info(coin_symbol)
        coin_symbol_link = coin_info["base_token_symbol"]
        buy_sell_link = generate_buy_sell_link(coin_symbol_link)
        processed_coin_data.append(
            {
                "email": data["email"],
                "tweet": data["tweet"],
                "tweet_url": data["url"],
                "twitter_handle": data["twitter_handle"],
                "coin_name": data["fetched_coin"],
                "coin_symbol": coin_symbol,
                "buy_sell_link": buy_sell_link,
                "url": data["url"],
                "coin_info": coin_info,
            }
        )
        with open("processed_coin_data.json", "a") as f:
            json.dump(processed_coin_data, f)
            f.write("\n")
    return processed_coin_data


# Function to send email via Postmark
def send_email(
    email, tweet_text, coin_name, twitter_handle, coin_info, buy_sell_link, tweet_url
):
    logger.info(f"Sending email to {email} about coin {coin_name}")

    # Load the email body from the index.html file
    email_body_path = os.path.join(os.getcwd(), "new_temp.html")

    with open(email_body_path, "r") as file:
        body_content = file.read()

    # Replace placeholders with actual values from coin_info, handling None values
    body_content = body_content.replace(
        "{{ base_token_symbol }}", str(coin_info.get("base_token_symbol", "N/A"))
    )
    body_content = body_content.replace(
        "{{ price_usd }}", str(coin_info.get("price_usd", "N/A"))
    )
    body_content = body_content.replace(
        "{{ price_change_h24 }}", str(coin_info.get("price_change_h24", "N/A"))
    )
    body_content = body_content.replace(
        "{{ market_cap }}", str(coin_info.get("market_cap", "N/A"))
    )
    body_content = body_content.replace(
        "{{ image_url }}",
        str(coin_info.get("image_url", "https://example.com/default_image.png")),
    )

    # Dynamically construct social media icons only if URL exists
    social_links = ""

    if coin_info.get("twitter_url"):
        social_links += f"""
            <a href="{coin_info['twitter_url']}"><img src="https://storage.googleapis.com/termix-prod/Xradar/t.png"
                alt="Twitter" /></a>
        """

    if coin_info.get("website_url"):
        social_links += f"""
            <a href="{coin_info['website_url']}">
                <img src="https://storage.googleapis.com/termix-prod/Xradar/w.png" alt="Website" />
            </a>
        """
        
    if coin_info.get("telegram_url"):
        social_links += f"""
            <a href="{coin_info['telegram_url']}">
                <img src="https://storage.googleapis.com/termix-prod/Xradar/tel.png" width="24.6" height="24" alt="Telegram" />
            </a>
        """

    # Insert the social links into the template
    body_content = body_content.replace("{{ social_links }}", social_links)

    # Handle buy/sell link and embedded tweet
    body_content = body_content.replace("{{ buy_sell_link }}", buy_sell_link)

    tweet_embed = fetch_tweet_embed(tweet_url)
    body_content = body_content.replace(
        "{{ tweet_embed }}", tweet_embed if tweet_embed else "<p>Tweet unavailable.</p>"
    )

    html_output_path = "coin_email_template_debug.html"
    with open(html_output_path, "w") as file:
        file.write(body_content)
    # Send the email
    try:
        postmark.emails.send(
            From="xradar@termix.ai",
            To=email,
            Subject=f"XRadar Predicted a Coin from Twitter Handle - {twitter_handle}",
            HtmlBody=body_content,
        )
        logger.info(f"Email successfully sent to {email}")
    except Exception as e:
        logger.error(f"Failed to send email to {email}: {e}")


# async def process_and_email():
#     logger.info("Processing tweets and sending emails...")
#     coins_data = await fetch_and_process_tweets()

#     if coins_data:
#         processed_coin_data = process_coins_data(coins_data)
#         for data in processed_coin_data:
#             send_email(data['email'], data['tweet'], data['coin_name'], data['twitter_handle'])
#     else:
#         logger.info("No coins found, resetting for the next cycle after 30 minutes.")


async def process_and_email():
    logger.info("Processing tweets and sending emails...")
    coins_data = await fetch_and_process_tweets()

    if coins_data:
        processed_coin_data = process_coins_data(coins_data)
        for data in processed_coin_data:
            # Pass additional arguments (coin_info, buy_sell_link, tweet_url) to send_email
            send_email(
                email=data["email"],
                tweet_text=data["tweet"],
                coin_name=data["coin_name"],
                twitter_handle=data["twitter_handle"],
                coin_info=data["coin_info"],  # Pass the entire coin_info dict
                buy_sell_link=data["buy_sell_link"],  # Buy/sell link
                tweet_url=data["url"],  # URL for the tweet
            )
    else:
        logger.info("No coins found, resetting for the next cycle after 30 minutes.")


# Schedule the process every 30 minutes using asyncio
async def schedule_process():
    while True:
        logger.info("Running the process...")
        try:
            await process_and_email()
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        await asyncio.sleep(cache_duration)


if __name__ == "__main__":
    asyncio.run(schedule_process())
