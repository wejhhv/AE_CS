from icrawler.builtin import BingImageCrawler

def Image_colleciton(search_keyword):
    
    crawler = BingImageCrawler(storage={"root_dir":"animal"})
    crawler.crawl(keyword=search_keyword, max_num=10)


#集めたい画像のキーワード
Image_colleciton("animal")


