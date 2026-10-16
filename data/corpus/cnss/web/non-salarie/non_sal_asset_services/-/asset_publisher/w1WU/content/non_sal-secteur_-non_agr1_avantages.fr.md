---
doc_id: cnss:/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal-secteur_-non_agr1_avantages#index
site: cnss
org: Caisse Nationale de Sécurité Sociale (CNSS)
trust: official
url: https://www.cnss.tn/fr/web/non-salarie/non_sal_asset_services/-/asset_publisher/w1WU/content/non_sal-secteur_-non_agr1_avantages
lang: fr
title: Indépendants et autres catégories - CNSS
section: null
breadcrumb:
- CNSS
- Non Salarié
- Services Non Salarié
- Non_Sal Secteur_ non_agr1_avantages
doc_type: procedure
domain_hint: employment
subtopic_hint: non-salaries
hint_score: 9
hint_keywords:
- non salarié
- indépendant
forms: []
render: html
discovered_via: link
source_updated: '2024-06-27'
http_last_modified: null
sitemap_lastmod: null
fetched_at: '2026-10-05T20:52:17+00:00'
first_seen: '2026-10-05T20:52:17+00:00'
last_changed: '2026-10-05T20:52:17+00:00'
content_hash: sha256:daf6109307f5665cc5f491de151f90f748354547d939090b9579672b78ae3871
words: 435
status: active
hint_source: llm
rag_exclude: true
rag_exclude_reason: server_error_page
---

- Une erreur s'est produite en traitant le calibre.

  ```
  Invocation of method 'parseInt' in  class java.lang.Integer threw exception java.lang.NumberFormatException: null at 64333136455[line 69, column 30]
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
  5#set ($categorie= $httpUtil.decodeURL($map.categorie.get(0), true))
  ```

  ```
  6#set ($secteur= $httpUtil.decodeURL($map.secteur.get(0), true))
  ```

  ```
  7#set ($champAppli= $httpUtil.decodeURL($map.champAppli.get(0), true))
  ```

  ```
  8#set ($servActif= $httpUtil.decodeURL($map.servActif.get(0), true))
  ```

  ```
  ```

  ```
  10<script>
  ```

  ```
  11var J=jQuery.noConflict();
  ```

  ```
  12J(function(){
  ```

  ```
  ```

  ```
  14if($lien.getData().length() > 100)
  ```

  ```
  15   J('#menu_pratique$avantage').find('ul').slideToggle('normal');
  ```

  ```
  16J('#menu_pratique$avantage').toggleClass('active');
  ```

  ```
  17});
  ```

  ```
  18function initMenusAvantagess() {
  ```

  ```
  19	J('ul.toggle_avantage li ul.smenu_avantage').hide();
  ```

  ```
  20	J('ul.toggle_avantage li a').click(
  ```

  ```
  21		function() {
  ```

  ```
  22			var checkElement = J(this).next();
  ```

  ```
  23			var parent = this.parentNode.parentNode.id;
  ```

  ```
  24			J('ul.toggle_avantage li ul.smenu_avantage').slideUp('normal');
  ```

  ```
  25			J('ul.toggle_avantage').removeClass('active');
  ```

  ```
  26			if(J('#' + parent).hasClass('noaccordion')) {
  ```

  ```
  27				J(this).next().slideToggle('normal');
  ```

  ```
  28				J('#' + parent).toggleClass('active');
  ```

  ```
  29				return false;
  ```

  ```
  30			}
  ```

  ```
  31		}
  ```

  ```
  32	);
  ```

  ```
  33}
  ```

  ```
  34</script>
  ```

  ```
  ```

  ```
  36<ul id="menu_pratique">
  ```

  ```
  37   <li>
  ```

  ```
  38   #set($AssetTagLocalService = $serviceLocator.findService("com.liferay.portlet.asset.service.AssetTagLocalService"))
  ```

  ```
  39   #set($JournalArticleLocalService = $serviceLocator.findService("com.liferay.portlet.journal.service.JournalArticleLocalService"))
  ```

  ```
  ```

  ```
  41   #set ($searchTag = "services$categorie$secteur$champAppli")
  ```

  ```
  42   #set($articles = $JournalArticleLocalService.getStructureArticles(54202,"136173"))
  ```

  ```
  43   #set ($languageId = $languageUtil.getLanguageId($locale))
  ```

  ```
  ```

  ```
  45   #foreach ( $article in $articles)
  ```

  ```
  46        #if ( $JournalArticleLocalService.isLatestVersion(54202,$article.articleId, $article.version))
  ```

  ```
  ```

  ```
  48           #set($articlePK = $article.resourcePrimKey)
  ```

  ```
  49	   #set($tagsAsset = $AssetTagLocalService.getTagNames("com.liferay.portlet.journal.model.JournalArticle", $articlePK))
  ```

  ```
  ```

  ```
  51           #foreach( $tag in $tagsAsset )
  ```

  ```
  52	      #if($tag == $searchTag)
  ```

  ```
  53                #set ($document = $saxReaderUtil.read($article.getContentByLocale($request.locale)))
  ```

  ```
  54		#set ($root = $document.getRootElement())
  ```

  ```
  55                #set ($titres= $root.selectNodes("dynamic-element[@name='services']"))
  ```

  ```
  ```

  ```
  57                #foreach ($titre in $titres)
  ```

  ```
  58                       #if($velocityCount == $servActif)
  ```

  ```
  59                              #set ($titre_content= $titre.selectSingleNode("dynamic-content"))
  ```

  ```
  60                              #set ($titre_content_article= $titre_content.text)
  ```

  ```
  61 	               #end
  ```

  ```
  62	        #end
  ```

  ```
  63                <h2> $titre_content_article </h2>
  ```

  ```
  64	     #end
  ```

  ```
  65           #end
  ```

  ```
  66        #end
  ```

  ```
  67    #end
  ```

  ```
  68    #set($Integer = 0)
  ```

  ```
  69#set($newAvantage = $Integer.parseInt($avantage) - 1)
  ```

  ```
  70    <ul class="tabs-left">
  ```

  ```
  71	                 #if($newAvantage.equals(0))
  ```

  ```
  72                            <li id="menu_pratique$param" class='active first'>
  ```

  ```
  73						<a href="/web/non-salarie/non_sal_asset_services_avantages/-/asset_publisher/w1WU/content/$lien.getSiblings().get(0).lien-article.getData()?convention=false&categorie=$categorie&secteur=$secteur&champAppli=$champAppli&servActif=$servActif&avantage=1">
  ```

  ```
  74                                           $lien.getSiblings().get(0).short-lien.getData()
  ```

  ```
  75				    </a>
  ```

  ```
  76			    </li>
  ```

  ```
  77			#end
  ```

  ```
  78                        #foreach($field1 in $lien.getSiblings())
  ```

  ```
  79                                #if($velocityCount == $newAvantage)
  ```

  ```
  80                                        <li id="menu_pratique$param" class='active first'>
  ```

  ```
  81						<a href="/web/non-salarie/non_sal_asset_services_avantages/-/asset_publisher/w1WU/content/$field1.lien-article.getData()?convention=false&categorie=$categorie&secteur=$secteur&champAppli=$champAppli&servActif=$servActif&avantage=$velocityCount">
  ```

  ```
  82							$lien.getSiblings().get($velocityCount ).short-lien.getData()
  ```

  ```
  83						</a>
  ```

  ```
  84					</li>
  ```

  ```
  85				#end
  ```

  ```
  86			#end
  ```

  ```
  87			#foreach($field1 in $lien.getSiblings())
  ```

  ```
  88				#if($velocityCount != $avantage)
  ```

  ```
  89					<li id="menu_pratique$velocityCount">
  ```

  ```
  90						<a href="/web/non-salarie/non_sal_asset_services_avantages/-/asset_publisher/w1WU/content/$field1.lien-article.getData()?convention=false&categorie=$categorie&secteur=$secteur&champAppli=$champAppli&servActif=$servActif&avantage=$velocityCount">
  ```

  ```
  91							$field1.short-lien.getData()
  ```

  ```
  92						</a>
  ```

  ```
  93					</li>
  ```

  ```
  94				#end
  ```

  ```
  95			#end
  ```

  ```
  96		</ul>
  ```

  ```
  97   </li>
  ```

  ```
  98</ul>
  ```
