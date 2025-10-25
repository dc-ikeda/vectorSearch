Class dev.ImageData Extends %Persistent [ SqlRowIdPrivate ]
{

Parameter USEEXTENTSET = 1;

Index DDLBEIndex [ Extent, SqlName = "%%DDLBEIndex", Type = bitmap ];

Index HNSWIndex On (imgvec) As %SQL.Index.HNSW(Distance = "Cosine") [ SqlName = HNSWIndex, Type = bitmap ];

Property url As %String(MAXLEN = 256) [ SqlColumnNumber = 2 ];

Property imgvec As %Vector(DATATYPE = "float", LEN = 512) [ SqlColumnNumber = 3 ];

Storage Default
{
<Data name="ImageDataDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>url</Value>
</Value>
</Data>
<Data name="_GBL_imgvec">
<Attribute>imgvec</Attribute>
<DataDefinitionLocation>^JobW.Cpdb.1.P1</DataDefinitionLocation>
<Structure>node</Structure>
<Value name="1">
<Value>imgvec</Value>
</Value>
</Data>
<DataLocation>^JobW.Cpdb.1</DataLocation>
<DefaultData>ImageDataDefaultData</DefaultData>
<ExtentLocation>^JobW.Cpdb</ExtentLocation>
<IdLocation>^JobW.Cpdb.1</IdLocation>
<Index name="DDLBEIndex">
<Location>^JobW.Cpdb.2</Location>
</Index>
<Index name="HNSWIndex">
<Location>^JobW.Cpdb.3</Location>
</Index>
<Index name="IDKEY">
<Location>^JobW.Cpdb.1</Location>
</Index>
<IndexLocation>^JobW.Cpdb.I</IndexLocation>
<StreamLocation>^JobW.Cpdb.S</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
