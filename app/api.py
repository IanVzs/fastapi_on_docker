import aiohttp
# TODO 还未写完, 还需要写吗?

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()
