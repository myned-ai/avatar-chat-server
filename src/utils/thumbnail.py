import re
import urllib.request
import urllib.error
import logging
import asyncio

logger = logging.getLogger(__name__)

# Basic regex to find og:image content
OG_IMAGE_REGEX = re.compile(r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
OG_IMAGE_REGEX_ALT = re.compile(r'<meta[^>]*content=[\'"]([^\'"]+)[\'"][^>]*property=[\'"]og:image[\'"]', re.IGNORECASE)

def _fetch_thumbnail_sync(url: str) -> str | None:
    """Synchronous core to fetch the thumbnail to run in a thread."""
    try:
        # Some servers reject default urllib User-Agent
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Try to find the og:image meta tag
            match = OG_IMAGE_REGEX.search(html) or OG_IMAGE_REGEX_ALT.search(html)
            if match:
                img_url = match.group(1)
                # Handle relative URLs if necessary
                if img_url.startswith('/'):
                    from urllib.parse import urljoin
                    img_url = urljoin(url, img_url)
                return img_url
    except Exception as e:
        logger.debug(f"Failed to fetch thumbnail for {url}: {e}")
    
    return None

async def get_link_thumbnail(url: str) -> str | None:
    """
    Asynchronously fetches the OpenGraph image (og:image) for a given URL.
    Returns the image URL if found, otherwise None.
    """
    if not url:
        return None
        
    loop = asyncio.get_running_loop()
    # Run in a threadpool to avoid blocking the async event loop with sync network i/o
    return await loop.run_in_executor(None, _fetch_thumbnail_sync, url)
