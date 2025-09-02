import pandas as pd
import random

# Load existing dataset
DATA_FILE = "twitter_sentiment_dataset.csv"
df = pd.read_csv(DATA_FILE)

# Define topics and sentiments
topics = ["geopolitics", "movies", "sports", "general"]
sentiments = ["happy", "sad", "fear", "angry", "confusion", "supportive", "opposing", "irrelevant"]

# Example tweets for each sentiment
examples = {
    "happy": [
        "The peace treaty brings hope for the region",
        "The new superhero film was absolutely amazing",
        "Our team’s victory parade was incredible",
        "I love how the community came together"
    ],
    "sad": [
        "The new sanctions will hurt the economy badly",
        "The ending of that drama made me cry",
        "We lost the finals again, so disappointing",
        "It’s heartbreaking to see this happen"
    ],
    "fear": [
        "The sudden military drills are frightening",
        "The horror trailer gave me chills",
        "The player’s injury is worrying",
        "I’m scared about the rising tensions"
    ],
    "angry": [
        "That speech by the minister made me furious",
        "That remake ruined the original classic",
        "The referee’s decision was outrageous",
        "I’m angry about the unfair treatment"
    ],
    "confusion": [
        "I’m confused about the new border agreement",
        "I’m not sure what the plot was about",
        "I don’t understand the new rules",
        "This situation makes no sense to me"
    ],
    "supportive": [
        "I fully support the climate accord",
        "I support more indie films getting recognition",
        "I support the coach’s decision",
        "I stand with the people affected"
    ],
    "opposing": [
        "I strongly oppose the trade restrictions",
        "I oppose the casting choice for the lead role",
        "I oppose the new tournament format",
        "I’m against this policy change"
    ],
    "irrelevant": [
        ";lkjhgfdsa qwertyuiop",
        "asdfghjkl zxcvbnm qwerty",
        "buy followers cheap $$$ click here",
        "lorem ipsum dolor sit amet",
        "1234567890 abcdefghijkl",
        "random words without meaning",
        "blue chair runs softly",
        "win free iPhone now!!!"
    ]
}

# Determine target per sentiment for perfect balance
total_target = 250
target_per_class = total_target // len(sentiments)  # 31 each, with 2 classes getting 32

# Count current per sentiment
current_counts = df['sentiment'].value_counts().to_dict()

# Fill missing counts
rows = []
next_id = df['id'].max() + 1

for sentiment in sentiments:
    needed = target_per_class - current_counts.get(sentiment, 0)
    if needed < 0:
        needed = 0  # Already has more than target
    for _ in range(needed):
        topic = random.choice(topics)
        tweet = random.choice(examples[sentiment])
        rows.append([next_id, topic, tweet, sentiment])
        next_id += 1

# Append and save
new_df = pd.DataFrame(rows, columns=["id", "topic", "tweet", "sentiment"])
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv(DATA_FILE, index=False)

print(f"✅ Added {len(new_df)} balanced rows. Final dataset size: {len(df)}")
print("Class distribution after balancing:")
print(df['sentiment'].value_counts())
