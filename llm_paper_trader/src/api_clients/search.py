from duckduckgo_search import DDGS

class SearchClient:
    def __init__(self, api_key: str = None):
        # API key is ignored, structural signature preserved for downstream compatibility
        self.api_key = api_key

    def search_news(self, query: str, max_results: int = 5) -> str:
        """
        Uses DuckDuckGo to fetch recent online news to serve as context for the LLM.
        Returns a formatted Markdown string completely locally and for free.
        """
        try:
            # We want recent, relevant information formatted nicely.
            results = DDGS().news(keywords=query, max_results=max_results)
            
            context = f"## DDG Search Results for: {query}\n\n"
            if not results:
                return context + "No recent news found."
                
            for row in results:
                context += f"- **{row.get('title', 'Unknown')}**: {row.get('body', '')}\n"
                
            return context
            
        except Exception as e:
            print(f"Search Extraction Error (DDG): {e}")
            return f"Error fetching search context: {e}"
