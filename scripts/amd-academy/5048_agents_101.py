"""AMD 5048 Agents 101：连接 vLLM、Pydantic AI 和时间/Airbnb MCP。

依赖：pydantic-ai-slim[mcp]==1.65.0、mcp-server-time、Node.js 22 和 npm。
先启动 OpenAI 兼容的 vLLM 服务；使用 --help 查看实验阶段和配置。
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/tmp/agents-101')
DEFAULT_PROMPT = 'Find a place to stay in Vancouver for next Sunday for 3 nights for 2 adults?'


def parse_args():
    parser = argparse.ArgumentParser(description='AMD 5048 Agents 101：vLLM + Pydantic AI + MCP')
    parser.add_argument('--stage', choices=('basic', 'local-time', 'mcp-time', 'airbnb', 'all'), default='all')
    parser.add_argument('--base-url', default=os.environ.get('BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--model', default='Qwen3.5-9B', help='vLLM 的 served model name')
    parser.add_argument('--timezone', default='America/New_York', help='时间 MCP 的 IANA 时区')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Airbnb 查询内容')
    parser.add_argument('--airbnb-dir', type=Path, default=ROOT / 'airbnb-mcp')
    parser.add_argument('--install-airbnb', action='store_true', help='用 npm 安装 notebook 指定的 Airbnb MCP 版本')
    parser.add_argument('--ignore-robots-txt', action='store_true', help='传递 notebook 中的 --ignore-robots-txt 参数')
    return parser.parse_args()


def prepare_airbnb(args):
    if not shutil.which('node') or (args.install_airbnb and not shutil.which('npm')):
        raise RuntimeError('需要 Node.js 22；安装 MCP 时还需要 npm。')
    if args.install_airbnb:
        args.airbnb_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['npm', 'install', '--prefix', str(args.airbnb_dir), '@openbnb/mcp-server-airbnb@0.3.0'], check=True)
    entry = args.airbnb_dir / 'node_modules/@openbnb/mcp-server-airbnb/dist/index.js'
    if not entry.is_file():
        raise RuntimeError(f'找不到 Airbnb MCP：{entry}；请使用 --install-airbnb 安装。')
    # notebook 将 Photon 请求超时从 5 秒调至 30 秒，避免网络较慢时失败。
    source = entry.read_text(encoding='utf-8')
    old = 'const timeoutId = setTimeout(() => controller.abort(), 5000);'
    new = 'const timeoutId = setTimeout(() => controller.abort(), 30000);'
    if source.count(old) + source.count(new) != 1:
        raise RuntimeError(f'Airbnb MCP 的超时代码与 notebook 不一致：{entry}')
    if old in source:
        entry.write_text(source.replace(old, new), encoding='utf-8')
    return entry


async def main_async(args):
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.environ.get('OPENAI_API_KEY', 'abc-123')
    model = OpenAIChatModel(args.model, provider=OpenAIProvider(base_url=args.base_url, api_key=api_key))

    def get_current_date() -> str:
        """Return the current local date and time."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    async def run(label, agent, prompt):
        print(f'\n--- {label} ---\n问题: {prompt}', flush=True)
        async with agent.run_mcp_servers():
            result = await agent.run(prompt)
        print(f'回答: {result.output}', flush=True)

    stages = ('basic', 'local-time', 'mcp-time', 'airbnb') if args.stage == 'all' else (args.stage,)
    if 'basic' in stages:
        agent = Agent(model=model)
        await run('基础问答', agent, 'What is the capital of France?')
        await run('无工具的日期问答', agent, 'What is the date today?')
    if 'local-time' in stages:
        agent = Agent(model=model, tools=[get_current_date], system_prompt='Use get_current_date for date/time questions.')
        await run('本地时间工具', agent, 'What is the date today?')
    if 'mcp-time' in stages or 'airbnb' in stages:
        # 启动本地MCP服务器
        # (1)基于python的MCP插件
        time_server = MCPServerStdio(sys.executable, args=['-m', 'mcp_server_time', f'--local-timezone={args.timezone}'])
        if 'mcp-time' in stages:
            agent = Agent(model=model, mcp_servers=[time_server], system_prompt='Use get_current_time for date/time questions.')
            await run('时间 MCP', agent, 'What is the date today?')
        if 'airbnb' in stages:
            entry = prepare_airbnb(args)
            node_args = [str(entry)] + (['--ignore-robots-txt'] if args.ignore_robots_txt else [])
            proxy_env = {k: v for k, v in os.environ.items() if k.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}}
            # (2)基于node的MCP插件
            airbnb = MCPServerStdio('node', args=node_args, timeout=60, env={**proxy_env, 'NODE_USE_ENV_PROXY': '1'})
            agent = Agent(model=model, mcp_servers=[time_server, airbnb], system_prompt=(
                'You have get_current_time, airbnb_search and airbnb_listing_details tools. '
                'For listings, check the current date first, then search. Report tool results accurately.'
            ))
            await run('Airbnb 查询', agent, args.prompt)


def main():
    args = parse_args()
    print(f'模型服务: {args.base_url}；模型: {args.model}；阶段: {args.stage}', flush=True)
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
