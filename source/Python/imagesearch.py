import sys

import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

args = sys.argv

contents_search = args[1]

txtModel = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
search_vector = txtModel.encode(contents_search).tolist()
print(len(search_vector))


username = '_system'
password = 'SYS'
hostname = 'localhost'
port = 1972
namespace = 'TESTAI'

CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
engine = create_engine(CONNECTION_STRING)

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 5 url, VECTOR_COSINE(imgvec, TO_VECTOR(:txtvec, float, 512)) as sim FROM dev.ImageData
            ORDER BY sim DESC
        """)

        results = conn.execute(sql, {'txtvec': str(search_vector)}).fetchall()

result_df = pd.DataFrame(results, columns=['sim', 'url'])
print(result_df)