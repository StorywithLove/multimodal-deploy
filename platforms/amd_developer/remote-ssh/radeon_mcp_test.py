import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command='/data/lcx/codex/radeon-gmail/venv/bin/python', args=['/data/lcx/codex/radeon-gmail/radeon_gmail_mcp.py'])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            print('tools:', [tool.name for tool in (await session.list_tools()).tools])
            print('result:', (await session.call_tool('radeon_pool_status', {})).content)

asyncio.run(main())
