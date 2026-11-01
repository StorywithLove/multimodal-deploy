"""AMD 5061 Agents 201：CrewAI + vLLM、LangGraph + Ollama、Google ADK + A2A。

参考 AMD/Google 多智能体教程的三智能体采购案例。此脚本把 Notebook 分成
burger-server、pizza-server、root、all 等运行阶段，便于在 tmux 中分别启动。

配套 shell 依次准备模型及教程数据、Python 环境，再启动服务和演示。
也可用 --stage 分别运行准备、检查或单个智能体。
"""

import argparse
import asyncio
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

DEFAULT_ROOT = Path("/root/tmp/agents-201")
UPSTREAM_REPO = "https://github.com/shailensobhee/google-adk-agentic-ai-tutorial.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMD 5061 Agents 201：vLLM + Ollama + A2A 多智能体练习"
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "check", "burger-server", "pizza-server", "test-burger", "test-pizza", "root", "all"),
        default="all",
        help="prepare 下载教程辅助文件；check 检查依赖；server 启动销售智能体；root 启动 ADK/Gradio；all 启动整套演示",
    )
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("AGENTS_201_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--vllm-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8088/v1"))
    parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "ollama_chat/qwen3.5:9b"))
    parser.add_argument("--host", default=os.environ.get("A2A_HOST", "0.0.0.0"), help="A2A 服务监听地址")
    parser.add_argument("--advertise-host", default=os.environ.get("A2A_ADVERTISE_HOST", "127.0.0.1"), help="写入 Agent Card 的可访问主机名")
    parser.add_argument("--burger-port", type=int, default=int(os.environ.get("BURGER_PORT", "10001")))
    parser.add_argument("--pizza-port", type=int, default=int(os.environ.get("PIZZA_PORT", "10000")))
    parser.add_argument("--gradio-port", type=int, default=int(os.environ.get("GRADIO_PORT", "8095")))
    parser.add_argument("--burger-auth", default=os.environ.get("BURGER_AGENT_AUTH", "burgeruser123:burgerpass123"))
    parser.add_argument("--pizza-auth", default=os.environ.get("PIZZA_AGENT_AUTH", "pizza123"))
    parser.add_argument("--prompt", default="I want to order 2 pepperoni pizzas and 1 classic cheeseburger.")
    parser.add_argument("--share", action="store_true", help="启用 Gradio 公网临时分享链接")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> Path:
    """Clone the tutorial's utility branch; dependencies remain user-managed."""
    args.root.mkdir(parents=True, exist_ok=True)
    repo_dir = args.root / "google-adk-agentic-ai-tutorial"
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", "utils"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--single-branch", "--branch", "utils", UPSTREAM_REPO, str(repo_dir)],
            check=True,
        )
    runtime_dir = args.root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_dir / "utils", runtime_dir / "utils", dirs_exist_ok=True)
    client = runtime_dir / "utils" / "client.py"
    source = client.read_text(encoding="utf-8")
    if '"timeout": 30,' not in source and '"timeout": 300,' not in source:
        raise RuntimeError(f"未找到预期的客户端超时配置：{client}")
    client.write_text(source.replace('"timeout": 30,', '"timeout": 300,'), encoding="utf-8")
    print(f"教程辅助文件：{runtime_dir / 'utils'}")
    print(f"依赖清单：{repo_dir / 'requirements' / 'requirements.txt'}")
    print("请按教程确认依赖版本后安装；脚本不会自动改动当前 Python 环境。")
    return repo_dir


def ensure_utils(root: Path) -> Path:
    repo_dir = root / "runtime"
    if not (repo_dir / "utils").is_dir():
        raise RuntimeError(f"找不到教程 utils：{repo_dir / 'utils'}；先运行 --stage prepare。")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return repo_dir


def check(args: argparse.Namespace) -> int:
    packages = {
        "crewai": "crewai",
        "langchain_ollama": "langchain-ollama",
        "langgraph": "langgraph",
        "google.adk": "google-adk",
        "litellm": "litellm",
        "gradio": "gradio",
        "httpx": "httpx",
    }
    def module_exists(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            return False

    missing = [label for module, label in packages.items() if not module_exists(module)]
    print(f"Python: {sys.executable}")
    print(f"A2A utils: {args.root / 'runtime' / 'utils'}")
    print(f"vLLM API: {args.vllm_url}  model={args.vllm_model}")
    print(f"Ollama API: {args.ollama_url}  model={args.ollama_model}")
    if missing:
        print("缺少 Python 依赖：" + ", ".join(missing))
        print("参考依赖清单：" + str(args.root / "google-adk-agentic-ai-tutorial" / "requirements" / "requirements.txt"))
    else:
        print("Python 依赖：已找到所需模块")
    unavailable = False
    for name, url in (("vLLM", args.vllm_url.rstrip("/") + "/models"), ("Ollama", args.ollama_url.rstrip("/") + "/api/tags")):
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                print(f"{name}: HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"{name}: 无法连接（{exc}）")
            unavailable = True
    missing_utils = not (args.root / "runtime" / "utils").is_dir()
    if missing_utils:
        print("未找到 A2A utils；先运行 --stage prepare。")
    return 1 if missing or missing_utils or unavailable else 0


def build_order_types():
    from typing import Literal
    from pydantic import BaseModel

    class ResponseFormat(BaseModel):
        status: Literal["input_required", "completed", "error"] = "input_required"
        message: str

    class OrderItem(BaseModel):
        name: str
        quantity: int
        price: int

    class Order(BaseModel):
        order_id: str
        status: str
        order_items: list[OrderItem]

    return ResponseFormat, OrderItem, Order


def build_burger_agent(args: argparse.Namespace):
    from crewai import Agent as CrewAgent
    from crewai import Crew, LLM, Process, Task
    from crewai.tools import tool

    ResponseFormat, OrderItem, Order = build_order_types()

    @tool("create_order")
    def create_burger_order(order_items: list[OrderItem]) -> str:
        """Create an order after the customer confirms its contents and total."""
        order = Order(order_id=str(uuid.uuid4()), status="created", order_items=order_items)
        print(f"汉堡订单已创建：{order.model_dump()}", flush=True)
        return f"Order {order.model_dump()} has been created"

    instruction = """You are a burger shop seller. Menu and prices (IDR):
- Classic Cheeseburger: 85K
- Double Cheeseburger: 110K
- Spicy Chicken Burger: 80K
- Spicy Cajun Burger: 85K
Answer only burger menu and order questions. Never invent prices. Before creating
an order, require the customer to confirm the items and total; then call create_order.
Return a concise status and message. Use input_required until confirmation is clear."""

    class BurgerSellerAgent:
        SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

        def invoke(self, query: str, session_id: str) -> dict[str, Any]:
            seller = CrewAgent(
                role="Burger Seller Agent",
                goal="Explain the burger menu and create confirmed orders.",
                backstory="A careful burger shop seller who never guesses prices.",
                verbose=False,
                allow_delegation=False,
                tools=[create_burger_order],
                llm=LLM(model=args.vllm_model, provider="openai", base_url=args.vllm_url, api_key=os.environ.get("OPENAI_API_KEY", "local")),
            )
            task = Task(
                description=instruction + f"\nCustomer request: {query}\nSession: {session_id}",
                expected_output="A structured status and customer-facing message.",
                output_pydantic=ResponseFormat,
                agent=seller,
            )
            response = Crew(agents=[seller], tasks=[task], process=Process.sequential, verbose=False).kickoff()
            result = response.pydantic
            if result and isinstance(result, ResponseFormat):
                return {"is_task_complete": result.status == "completed", "require_user_input": result.status != "completed", "content": result.message}
            return {"is_task_complete": False, "require_user_input": True, "content": str(response)}

    return BurgerSellerAgent()


def build_pizza_agent(args: argparse.Namespace):
    from langchain_core.tools import tool
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    ResponseFormat, OrderItem, Order = build_order_types()
    memory = MemorySaver()

    @tool
    def create_pizza_order(order_items: list[OrderItem]) -> str:
        """Create a pizza order after the customer confirms items and total."""
        order = Order(order_id=str(uuid.uuid4()), status="created", order_items=order_items)
        print(f"披萨订单已创建：{order.model_dump()}", flush=True)
        return f"Order {order.model_dump()} has been created"

    instruction = """You are a pizza shop seller. Menu and prices (IDR):
- Margherita 100K; Pepperoni 140K; Hawaiian 110K; Veggie 100K; BBQ Chicken 130K.
Answer only pizza menu and order questions. Never invent prices. Require confirmation
of items and total before calling create_pizza_order. Return status and message."""

    class PizzaSellerAgent:
        SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

        def __init__(self) -> None:
            self.graph = create_react_agent(
                ChatOllama(model=args.ollama_model.split("/", 1)[-1], base_url=args.ollama_url),
                tools=[create_pizza_order],
                checkpointer=memory,
                prompt=instruction,
                response_format=ResponseFormat,
            )

        def invoke(self, query: str, session_id: str) -> dict[str, Any]:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.invoke({"messages": [("user", query)]}, config)
            result = state.get("structured_response")
            if result and isinstance(result, ResponseFormat):
                return {"is_task_complete": result.status == "completed", "require_user_input": result.status != "completed", "content": result.message}
            return {"is_task_complete": False, "require_user_input": True, "content": str(result or state)}

    return PizzaSellerAgent()


def start_seller_server(args: argparse.Namespace, seller: str) -> None:
    ensure_utils(args.root)
    from utils.server import A2AServer
    from utils.a2a_types import AgentAuthentication, AgentCapabilities, AgentCard, AgentSkill
    from utils.push_notification_auth import PushNotificationSenderAuth
    from utils.task_manager import AgentTaskManager

    if seller == "burger":
        agent = build_burger_agent(args)
        name, label, skill_id, port = "burger_seller_agent", "Burger Order Creation Tool", "create_burger_order", args.burger_port
        auth_scheme, auth_value = "Basic", args.burger_auth
        api_key = None
    else:
        agent = build_pizza_agent(args)
        name, label, skill_id, port = "pizza_seller_agent", "Pizza Order Creation Tool", "create_pizza_order", args.pizza_port
        auth_scheme, auth_value = "Bearer", args.pizza_auth
        api_key = args.pizza_auth

    address = f"http://{args.advertise_host}:{port}"
    card = AgentCard(
        name=name,
        description=f"Helps with {seller} menu questions and confirmed orders",
        url=address + "/",
        version="1.0.0",
        authentication=AgentAuthentication(schemes=[auth_scheme]),
        defaultInputModes=agent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=agent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(pushNotifications=True),
        skills=[AgentSkill(id=skill_id, name=label, description=label, tags=[f"{seller} order"], examples=[f"Show me the {seller} menu"])],
    )
    notification_auth = PushNotificationSenderAuth()
    notification_auth.generate_jwk()
    server_args = dict(
        agent_card=card,
        task_manager=AgentTaskManager(agent=agent, notification_sender_auth=notification_auth),
        host=args.host,
        port=port,
    )
    if seller == "burger":
        username, _, password = auth_value.partition(":")
        server_args.update(auth_username=username, auth_password=password)
    else:
        server_args["api_key"] = api_key
    server = A2AServer(**server_args)
    server.app.add_route("/.well-known/jwks.json", notification_auth.handle_jwks_endpoint, methods=["GET"])
    print(f"启动 {seller} A2A 服务：{address}", flush=True)
    server.start()


def test_seller(args: argparse.Namespace, seller: str) -> None:
    agent = build_burger_agent(args) if seller == "burger" else build_pizza_agent(args)
    prompt = "Show the menu and prices." if seller == "burger" else "Show the menu and prices."
    print(agent.invoke(prompt, f"test-{seller}-{uuid.uuid4()}") , flush=True)


def build_purchasing_agent(args: argparse.Namespace):
    ensure_utils(args.root)
    import litellm
    from google.adk import Agent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools.tool_context import ToolContext
    from utils.a2a_types import (
        AgentCard, Message, Part, Task, TaskArtifactUpdateEvent,
        TaskSendParams, TaskState, TaskStatusUpdateEvent, TextPart,
    )
    from utils.card_resolver import A2ACardResolver
    from utils.client import A2AClient

    os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    litellm.register_model({args.ollama_model: {"supports_function_calling": True}})
    TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]
    auth_by_name = {"pizza_seller_agent": args.pizza_auth, "burger_seller_agent": args.burger_auth}

    class RemoteAgentConnection:
        def __init__(self, card: AgentCard, url: str):
            self.card = card
            self.client = A2AClient(card, auth=auth_by_name.get(card.name, ""), agent_url=url)

        async def send_task(self, request: TaskSendParams, callback: TaskUpdateCallback | None):
            response = await self.client.send_task(request.model_dump())
            task = response.result
            if callback:
                callback(task, self.card)
            return task

    class PurchasingAgent:
        def __init__(self):
            self.connections = {}
            cards = []
            for url in (f"http://{args.advertise_host}:{args.pizza_port}", f"http://{args.advertise_host}:{args.burger_port}"):
                card = A2ACardResolver(url).get_agent_card()
                self.connections[card.name] = RemoteAgentConnection(card, url)
                cards.append({"name": card.name, "description": card.description})
            self.agent_list = "\n".join(str(card) for card in cards)

        def instruction(self, context: ReadonlyContext) -> str:
            return f"""You coordinate a pizza seller and a burger seller through A2A.
For each item, delegate only to the matching seller using send_task. Do not invent
menu items, prices, or order confirmations. Relay any confirmation question to the
user. Combine the sellers' actual responses clearly. Available agents: {self.agent_list}"""

        def before_model(self, context: CallbackContext, request):
            context.state.setdefault("session_id", str(uuid.uuid4()))

        def create(self):
            return Agent(
                model=LiteLlm(model=args.ollama_model),
                name="purchasing_agent",
                description="Routes food purchase requests to specialist sellers over A2A.",
                instruction=self.instruction,
                before_model_callback=self.before_model,
                tools=[self.send_task],
            )

        async def send_task(self, agent_name: str, task_text: str, tool_context: ToolContext):
            connection = self.connections.get(agent_name)
            if connection is None:
                raise ValueError(f"Unknown seller agent: {agent_name}")
            state = tool_context.state
            session_id = state.setdefault("session_id", str(uuid.uuid4()))
            request = TaskSendParams(
                id=str(uuid.uuid4()),
                sessionId=session_id,
                message=Message(role="user", parts=[TextPart(text=task_text)], metadata={"conversation_id": session_id}),
                acceptedOutputModes=["text", "text/plain"],
                metadata={"conversation_id": session_id},
            )
            task = await connection.send_task(request, None)
            if task.status.state == TaskState.INPUT_REQUIRED:
                tool_context.actions.escalate = True
            texts = []
            if task.status.message:
                texts.extend(part.text for part in task.status.message.parts if getattr(part, "type", None) == "text")
            for artifact in task.artifacts or []:
                texts.extend(part.text for part in artifact.parts if getattr(part, "type", None) == "text")
            return "\n".join(texts) or f"Remote task status: {task.status.state}"

    return PurchasingAgent().create()


async def run_root(args: argparse.Namespace) -> None:
    import gradio as gr
    from google.adk.events import Event
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    agent = build_purchasing_agent(args)
    app_name, user_id, session_id = "purchasing_concierge_app", "default_user", str(uuid.uuid4())
    sessions = InMemorySessionService()
    await sessions.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=sessions)

    async def respond(message: str, history: list[dict[str, Any]]):
        del history
        events: Any = runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        )
        answers = []
        async for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        answers.append(part.text)
            if event.is_final_response():
                break
        return "\n".join(answers) or "No text response was returned."

    app = gr.ChatInterface(
        respond,
        title="Purchasing Concierge · AMD 5061 Agents 201",
        description="Google ADK delegates to CrewAI and LangGraph seller agents over A2A.",
    )
    print(f"Gradio UI: http://{args.advertise_host}:{args.gradio_port}", flush=True)
    app.launch(server_name=args.host, server_port=args.gradio_port, share=args.share)


def wait_for_url(url: str, process: subprocess.Popen, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"A2A server exited early (code {process.returncode}).")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception as exc:  # service may still be starting
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"等待 A2A 服务超时：{url}；最后错误：{last_error}")


def run_all(args: argparse.Namespace) -> None:
    children: list[subprocess.Popen] = []
    common = [
        "--root", str(args.root), "--vllm-url", args.vllm_url, "--vllm-model", args.vllm_model,
        "--ollama-url", args.ollama_url, "--ollama-model", args.ollama_model,
        "--host", args.host, "--advertise-host", args.advertise_host,
        "--burger-port", str(args.burger_port), "--pizza-port", str(args.pizza_port),
        "--burger-auth", args.burger_auth, "--pizza-auth", args.pizza_auth,
    ]
    try:
        for role in ("burger", "pizza"):
            children.append(subprocess.Popen([sys.executable, __file__, "--stage", f"{role}-server", *common]))
        wait_for_url(f"http://{args.advertise_host}:{args.burger_port}/.well-known/agent.json", children[0])
        wait_for_url(f"http://{args.advertise_host}:{args.pizza_port}/.well-known/agent.json", children[1])
        print("两个 A2A 销售智能体已就绪，启动采购管家…", flush=True)
        asyncio.run(run_root(args))
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    if args.stage == "prepare":
        prepare(args)
        return 0
    if args.stage == "check":
        return check(args)
    if args.stage == "burger-server":
        start_seller_server(args, "burger")
    elif args.stage == "pizza-server":
        start_seller_server(args, "pizza")
    elif args.stage == "test-burger":
        test_seller(args, "burger")
    elif args.stage == "test-pizza":
        test_seller(args, "pizza")
    elif args.stage == "root":
        asyncio.run(run_root(args))
    elif args.stage == "all":
        run_all(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已停止。", flush=True)
        raise SystemExit(130)
