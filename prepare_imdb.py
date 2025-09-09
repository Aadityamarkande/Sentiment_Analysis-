import os
import pandas as pd

def load_imdb_data(folder_path):
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(folder_path, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return pd.DataFrame({'Sentence': texts, 'Sentiment': labels})

# Update path below to where you extracted aclImdb/train
train_folder = r'G:\Sentiment_Analysis\Sentiment-Analysis\aclImdb\train'  # Replace with your actual path

print("Loading IMDB training data...")
imdb_train_df = load_imdb_data(train_folder)
print(f"Loaded {len(imdb_train_df)} samples.")

# Save to CSV for training
imdb_train_df.to_csv("imdb_train.csv", index=False)
print("Saved imdb_train.csv")
