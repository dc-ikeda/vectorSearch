from pyarrow.parquet import ParquetFile
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from PIL import Image
import requests

pf = ParquetFile(r"D:\Python\image\recruit-jp.parquet")
#first_rows = next(pf.iter_batches(batch_size = 50))
first_rows = next(pf.iter_batches())
df = pa.Table.from_batches([first_rows]).to_pandas()
df = df.replace("\n", "", regex=True)

print(df.head(5)) # id, license, license_urll, url, category


# 画像化
def load_image(url_or_path):
    try:
        urls = url_or_path.split('_o')
        newUrl = urls[0] + '_b' +  urls[1] # #newUrl = urls[0] + '_c' +  urls[1]

        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return Image.open(requests.get(newUrl, stream=True).raw)
        else:
            return Image.open(url_or_path)
    except Exception as e:
        print(repr(e) +":"+ url_or_path)
        return ''

imgModel = SentenceTransformer('clip-ViT-B-32')

# images = [load_image(img) for img in df['url']]
images = []
urlList = []
imgVec = []
count = 0
for img in df['url']:
    count += 1
    imgObj = load_image(img)
    if not imgObj == '':
        images.append(imgObj)
        urlList.append(img)
    else:
        print(str(count))

embeddings = imgModel.encode(images)
imgVec = embeddings.tolist()

print(df.head(5))


# ------------------------------------------------
username = '_system'
password = 'SYS'
hostname = 'localhost'
port = 1972
namespace = 'TESTAI'

CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
engine = create_engine(CONNECTION_STRING)

with engine.connect() as conn:
    with conn.begin():
        for vec, url in zip(imgVec, urlList):
            sql = text(
                """
                    INSERT INTO dev.ImageData (url, imgvec)
                    VALUES (:url, TO_VECTOR(:imgvec))
                """
            )
            conn.execute(sql, {
                'url': url,
                'imgvec': str(vec)
            })
