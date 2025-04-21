from typing import List, Dict, Literal
from pydantic import BaseModel, Field

import asyncio
from aiohttp import ClientSession, ClientError

from bs4 import BeautifulSoup
import pdfplumber, io

class ConsolidatedDoc(BaseModel):
    url: str = Field(description="Link to the document")
    page_content: str = Field(description="Document content")
    doc_type: Literal['html', 'pdf', 'error', 'unsupported', 'client_error', 'timeout'] = Field(description="Type of document")

class ConsolidateDocs:
    
    def __init__(self, doc_list: List[str], 
                 max_concurrency: int = 3,
                 user_agent: str = 'Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version'
                ):
        self.doc_list = doc_list
        self.max_concurrency = max_concurrency
        self.user_agent = user_agent

    def extract_body_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        body_tag = soup.find('body')
        if body_tag:
            return body_tag.get_text(separator='\n', strip=True)
        return ''

    def extract_pdf_content(self, pdf_bytes):
        with io.BytesIO(pdf_bytes) as pdf_file:  # Convert bytes to file-like object
            with pdfplumber.open(pdf_file) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
        return text.strip()

    async def fetch_url(self, session, url):
        try:
            async with session.get(url, 
                                   headers={"User-Agent": self.user_agent}) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    if 'application/pdf' in content_type or 'pdf' in url:
                        # Handle PDF data
                        pdf_bytes = await response.read()
                        text_content = self.extract_pdf_content(pdf_bytes)
                        return {"page_content": text_content, "url": url, "doc_type": "pdf"}
                    elif 'text/html' in content_type:
                        # Handle HTML data
                        html_content = await response.text()
                        body_text = self.extract_body_content(html_content)
                        return {"page_content": body_text, "url": url, "doc_type": "html"}
                    else:
                        print(f"Unsupported content type: {content_type} for URL: {url}")
                        return {"page_content": None, "url": url, "doc_type": "unsupported"}
                else:
                    print(f"Failed to load {url} with status code: {response.status}")
                    return {"page_content": None, "url": url, "doc_type": "error"}
        except asyncio.TimeoutError:
            print(f"Timeout occurred while loading {url}")
            return {"page_content": None, "url": url, "doc_type": "timeout"}
        except ClientError as e:
            print(f"Client error for {url}: {e}")
            return {"page_content": None, "url": url, "doc_type": "client_error"}

    async def load_docs(self) -> List[ConsolidatedDoc]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async with ClientSession() as session:
            tasks = [self.fetch_url_with_semaphore(session, url, semaphore) for url in self.doc_list]
            results = await asyncio.gather(*tasks)

            # Filter out None values (failed loads)
            successful_loads = [doc for doc in results if doc['doc_type'] in ['html', 'pdf']]
    
            # Process the loaded documents here
            print(f"Successfully loaded {len(successful_loads)} documents.")
            
            return [ConsolidatedDoc(**doc) for doc in successful_loads]

    async def fetch_url_with_semaphore(self, session, url, semaphore):
        async with semaphore:
            return await self.fetch_url(session, url)