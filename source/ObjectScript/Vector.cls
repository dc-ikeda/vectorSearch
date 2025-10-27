Class dev.Vector Extends %RegisteredObject
{

XData %import [ MimeType = application/python ]
{
import iris
import gc
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import datetime
}

ClassMethod makeData(count As %Integer = 100000) [ Language = python ]
{
    pf = ParquetFile(r"D:/Python/cc100_Parquet/cc100-ja_sharded.parquet")
    first_rows = next(pf.iter_batches(batch_size = count))
    df = pa.Table.from_batches([first_rows]).to_pandas()
    df = df.replace("\n", "", regex=True)

    for index,row in df.iterrows():
        txt = row['text']
        iris.cls('dev.Vector').saveData(txt)

        gc.collect()
}

ClassMethod saveData(text As %String)
{
    s cnvTxt = $tr(..Convert(text), "[]")

    &sql(insert into dev.SearchData (txt, vec) values(:text, TO_VECTOR(:cnvTxt, double, 768)))
}

ClassMethod Search(txt As %String = "")
{
    q:($g(txt)="")

    s txt = ..Convert(txt)
    s txt = $tr(txt, "[]")

    s query(1) = "select top 5 VECTOR_DOT_PRODUCT(contents_vector, TO_VECTOR(?, DOUBLE, 768)) as sim, contents"
    , query(2) = "from vectortest order by sim desc"
    , query = 2

    w !,"実行"

    s rset = ##class(%SQL.Statement).%ExecDirect(.stmt, .query, .txt)
    while rset.%Next() {
        w !,rset.%Get("contents")
    }
}

ClassMethod Convert(text As %String) As %String [ Language = python ]
{
    model = SentenceTransformer('stsb-xlm-r-multilingual')
    search_vector = model.encode(text, normalize_embeddings=True).tolist()
    return str(search_vector)
}

ClassMethod SearchImg(txt As %String) As %DynamicArray
{
    q:($g(txt)="")

    s txt = ..ConvertImg(txt)
    s txt = $tr(txt, "[]")

    s query(1) = "SELECT TOP 5 url, VECTOR_COSINE(imgvec, TO_VECTOR(?, float, 512)) as sim"
    , query(2) = "FROM dev.ImageData order by sim desc"
    , query = 2

    s ary = []
    s rset = ##class(%SQL.Statement).%ExecDirect(.stmt, .query, .txt)
    while rset.%Next() {
        s url = $replace(rset.%Get("url"),"_o.jpg","_b.jpg")
        d ary.%Push({
            "imgid":(url),
            "sim":(rset.%Get("sim"))
        })
    }
    q ary
}

ClassMethod ConvertImg(txt As %String) As %String [ Language = python ]
{
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    text_embeddings = text_model.encode(txt).tolist()
    return str(text_embeddings)
}

}

