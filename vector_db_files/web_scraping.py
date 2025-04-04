import requests
from langchain_community.document_loaders import TextLoader
from read_write import write_file
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

def do(url):
    try:
        print(f'url = {url}')
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = set()
            content = []
            for a_tag in soup.find_all('a', href=True):
                full_link = urljoin(url, a_tag['href'])
                result = urlparse(full_link)
                if all([result.scheme, result.netloc]):
                    links.add(full_link)
            for link in links:
                try:
                    inner_response = requests.get(link)
                    if inner_response.status_code == 200:
                        inner_soup = BeautifulSoup(inner_response.content, 'html.parser')
                        raw_text = inner_soup.get_text()
                        cleaned_text = " ".join(raw_text.split())  
                        content.append(cleaned_text)
                except Exception as e:
                    print(f"Error scraping inner loop {link}: {e}") 
            return content
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")

def scrape_and_write_to_file(links, base_dir):
    print(f"entering try")
    try:
        for idx, link in enumerate(links):
            content = do(link)
            file_name = os.path.join(base_dir, f'craw{idx+1}.md')
            cleaned_content = "\n".join(content)
            write_file(file_name, cleaned_content)
            print(f"Content saved to {file_name}")
    except Exception as e:
        print(f"Error scraping {link}: {e}")

def prepare_documents(directory):
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.md'):
            loader = TextLoader(os.path.join(directory, file_name))
            documents.extend(loader.load())
    return documents

def main():
    link = ['https://www.globaldata.com/company-profile/celanese-corp/']
    scrape_and_write_to_file(link, base)
    doccument = prepare_documents(base)
    print(doccument)

if __name__ == "__main__":
    base = '/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/vector_db_files/scraped_data'
    main()