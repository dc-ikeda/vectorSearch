import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# 引数
args = sys.argv
contents_search = args[1]

model = SentenceTransformer('stsb-xlm-r-multilingual')
search_vector = model.encode(contents_search, normalize_embeddings=True).tolist()


username = '_system'
password = 'SYS'
hostname = 'localhost'
port = 1972
namespace = 'USER'

CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
engine = create_engine(CONNECTION_STRING)

with engine.connect() as conn:
    with conn.begin():
        sql = text("""
            SELECT TOP 5 contents, VECTOR_DOT_PRODUCT(contents_vector, TO_VECTOR(:search_vector, double, 768)) as sim FROM vectortest
            ORDER BY sim DESC
        """)

        results = conn.execute(sql, {'search_vector': str(search_vector)}).fetchall()

result_df = pd.DataFrame(results, columns=['contents', 'sim'])
pd.set_option('display.max_colwidth', None)
print(result_df)