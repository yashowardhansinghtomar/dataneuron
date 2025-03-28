from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_csv('DataNeuron_Text_Similarity.csv')

# Define similarity function
def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

# Apply similarity to dataframe
df['similarity_score'] = df.apply(lambda row: compute_similarity(row['text1'], row['text2']), axis=1)

print(df.head())
