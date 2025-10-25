Class dev.Rest Extends %CSP.REST
{

Parameter CONTENTTYPE = "application/json";

Parameter HandleCorsRequest = 1;

Parameter UseSession = 0;

XData UrlMap
{
<Routes>
<Route Url="/sample" Method="GET" Call="GetImage" Cors="true"/>
<Route Url="/test" Method="GET" Call="Test" Cors="true"/>
</Routes>
}

ClassMethod GetImage() As %Status
{
	s search = $g(%request.Data("searchTxt",1))

    s start = $zh
    , getImgs = ##class(dev.Vector).SearchImg(search)
    , end = $zh
	d ##class(%REST.Impl).%WriteResponse({
        "image": (getImgs),
        "time" : (end-start)
    })
    q $$$OK
}

ClassMethod Test()
{
    d ##class(%REST.Impl).%WriteResponse({
		"SessionId": (%session.SessionId),
		"license"  : (%session.LicenseId)
	})
    q $$$OK
}

}
