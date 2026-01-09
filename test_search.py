#!/usr/bin/env python3
"""Test script for DuckDuckGo search."""
import asyncio
from duckduckgo_search import DDGS

async def test_search():
    """Test async search."""
    results = []
    try:
        # Run directly (not in executor)
        with DDGS() as ddgs:
            search_results = list(ddgs.text("artificial intelligence trends", max_results=5))
            print(f"Raw results: {len(search_results)}")
            for result in search_results:
                if result:
                    print(f"Result: {result}")
                    results.append({
                        "url": result.get("href", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "")
                    })
        # Yield to event loop
        await asyncio.sleep(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nFinal results: {len(results)}")
    for r in results:
        print(f"  - {r['title']}: {r['url']}")

if __name__ == "__main__":
    asyncio.run(test_search())
