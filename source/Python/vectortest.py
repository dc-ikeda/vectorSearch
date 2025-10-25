import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from pyarrow.parquet import ParquetFile
import pyarrow as pa

pf = ParquetFile(r"D:\Python\cc100_Parquet\cc100-ja_sharded.parquet")
first_rows = next(pf.iter_batches(batch_size = 100000))
df = pa.Table.from_batches([first_rows]).to_pandas()
df = df.replace("\n", "", regex=True)


model = SentenceTransformer('stsb-xlm-r-multilingual')
embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True)
df['text_vector'] = embeddings.tolist()


username = '_system'
password = 'SYS'
hostname = 'localhost'
port = 1972
namespace = 'USER'

CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
engine = create_engine(CONNECTION_STRING)

# テーブル作成
with engine.connect() as conn:
    with conn.begin():
        sql = f"""
            CREATE TABLE vectortest(
                contents VARCHAR(4096),
                contents_vector VECTOR(DOUBLE, 768)
            )
            """
        result = conn.execute( text(sql) )

# データ作成
with engine.connect() as conn:
    with conn.begin():
        for index, row in df.iterrows():
            sql = text(
                """
                    INSERT INTO vectortest
                    (contents, contents_vector)
                    VALUES (:contents, TO_VECTOR(:contents_vector))
                """
            )
            conn.execute(sql, {
                'contents': row['text'],
                'contents_vector': str(row['text_vector'])
            })