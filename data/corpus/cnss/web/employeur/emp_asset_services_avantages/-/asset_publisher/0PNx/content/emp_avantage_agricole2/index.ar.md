---
doc_id: cnss:/web/employeur/emp_asset_services_avantages/-/asset_publisher/0PNx/content/emp_avantage_agricole2#index
site: cnss
org: Caisse Nationale de Sécurité Sociale (CNSS)
trust: official
url: https://www.cnss.tn/ar/web/employeur/emp_asset_services_avantages/-/asset_publisher/0PNx/content/emp_avantage_agricole2
lang: ar
title: المؤجرون - CNSS
section: null
breadcrumb:
- CNSS
- Employeur
- Services Employeur Avantages
- Emp_Avantage_agricole2
doc_type: procedure
domain_hint: employment
subtopic_hint: affiliation-cnss
hint_score: 9
hint_keywords:
- employeur
forms: []
render: html
discovered_via: link
source_updated: '2019-06-18'
http_last_modified: null
sitemap_lastmod: null
fetched_at: '2026-10-05T20:47:05+00:00'
first_seen: '2026-10-05T20:47:05+00:00'
last_changed: '2026-10-05T20:47:05+00:00'
content_hash: sha256:4dbe5b27874e650a2040a9a269e274cfbd2694fc4c5be1d9b3d5505f80980f9b
words: 223
status: active
---

An error occurred while processing the template.

```
Invocation of method 'parseInt' in  class java.lang.Integer threw exception java.lang.NumberFormatException: null at 43234133127[line 13, column 30]
```

```
1#set ($currenturl = ($request.attributes.CURRENT_URL))
```

```
2#set ($qstring= $httpUtil.getQueryString($currenturl))
```

```
3#set ($map= $httpUtil.getParameterMap($qstring))
```

```
4#set ($avantage= $httpUtil.decodeURL($map.avantage.get(0), true))
```

```
5#set ($tagParam= $httpUtil.decodeURL($map.tagParam.get(0), true))
```

```
6#set ($secteur= $httpUtil.decodeURL($map.secteur.get(0), true))
```

```
```

```
8<script>
```

```
9var J=jQuery.noConflict();
```

```
10</script>
```

```
```

```
12#set($Integer = 0)
```

```
13#set($newAvantage = $Integer.parseInt($avantage) - 1)
```

```
```

```
15<ul id="menu_pratique">
```

```
16	<li>
```

```
17		<h2>$services.getSiblings().get(3).getData()</h2>
```

```
18		<ul class="tabs-left">
```

```
19			#if($newAvantage.equals(0))
```

```
20                <li id="menu_pratique$param" class='active first'>
```

```
21				    <a class="ui-icon" href="/web/employeur/emp_asset_services_avantages/-/asset_publisher/0PNx/content/$lien.getSiblings().get(0).lien-article.getData()?secteur=$secteur&tagParam=$tagParam&avantage=1">
```

```
22                        $lien.getSiblings().get(0).short-lien.getData()
```

```
23				    </a>
```

```
24			    </li>
```

```
25			#end
```

```
26            #foreach($field1 in $lien.getSiblings())
```

```
27				#if($velocityCount == $newAvantage)
```

```
28                    <li id="menu_pratique$param" class='active first'>
```

```
29						<a class="ui-icon" href="/web/employeur/emp_asset_services_avantages/-/asset_publisher/0PNx/content/$field1.lien-article.getData()?secteur=$secteur&tagParam=$tagParam&avantage=$velocityCount">
```

```
30                            $lien.getSiblings().get($velocityCount).short-lien.getData()
```

```
31						</a>
```

```
32					</li>
```

```
33				#end
```

```
34			#end
```

```
```

```
36			#foreach($field1 in $lien.getSiblings())
```

```
37				#if($velocityCount != $avantage)
```

```
38					<li id="menu_pratique$velocityCount">
```

```
39						<a class="ui-icon" href="/web/employeur/emp_asset_services_avantages/-/asset_publisher/0PNx/content/$field1.lien-article.getData()?secteur=$secteur&tagParam=$tagParam&avantage=$velocityCount">
```

```
40							$field1.short-lien.getData()
```

```
41						</a>
```

```
42					</li>
```

```
43				#end
```

```
44			#end
```

```
45		</ul>
```

```
46	</li>
```

```
47</ul>
```
