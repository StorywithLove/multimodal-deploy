import sys
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

import litellm

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "ollama_chat/qwen35")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PIZZA_SELLER_AGENT_URL = os.getenv("PIZZA_AGENT_URL", "http://localhost:10000")
PIZZA_SELLER_AGENT_AUTH = os.getenv("PIZZA_AGENT_AUTH", "pizza123")
BURGER_SELLER_AGENT_URL = os.getenv("BURGER_AGENT_URL", "http://localhost:10001")
BURGER_SELLER_AGENT_AUTH = (
    f"{os.getenv('BURGER_AGENT_USER', 'burgeruser123')}:"
    f"{os.getenv('BURGER_AGENT_PASSWORD', 'burgerpass123')}"
)

litellm.register_model({OLLAMA_CHAT_MODEL: {"supports_function_calling": True}})

from typing import Callable
import uuid
from utils.a2a_types import (
    AgentCard,
    Task,
    TaskSendParams,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from utils.client import A2AClient
import os

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]

KNOWN_AUTH = {
    "pizza_seller_agent": PIZZA_SELLER_AGENT_AUTH,
    "burger_seller_agent": BURGER_SELLER_AGENT_AUTH,
}


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        auth = KNOWN_AUTH.get(agent_card.name, None)
        self.agent_client = A2AClient(agent_card, auth=auth, agent_url=agent_url)
        self.card = agent_card

        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_task(
        self,
        request: TaskSendParams,
        task_callback: TaskUpdateCallback | None,
    ) -> Task | None:
        response = await self.agent_client.send_task(request.model_dump())
        merge_metadata(response.result, request)
        # For task status updates, we need to propagate metadata and provide
        # a unique message id.
        if (
            hasattr(response.result, "status")
            and hasattr(response.result.status, "message")
            and response.result.status.message
        ):
            merge_metadata(response.result.status.message, request.message)
            m = response.result.status.message
            if not m.metadata:
                m.metadata = {}
            if "message_id" in m.metadata:
                m.metadata["last_message_id"] = m.metadata["message_id"]
            m.metadata["message_id"] = str(uuid.uuid4())

        if task_callback:
            task_callback(response.result, self.card)
        return response.result


def merge_metadata(target, source):
    if not hasattr(target, "metadata") or not hasattr(source, "metadata"):
        return
    if target.metadata and source.metadata:
        target.metadata.update(source.metadata)
    elif source.metadata:
        target.metadata = dict(**source.metadata)

import json
import uuid
from typing import List
import httpx
import os

from google.adk.models.lite_llm import LiteLlm 
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from utils.card_resolver import A2ACardResolver
from utils.a2a_types import (
    AgentCard,
    Message,
    TaskState,
    Task,
    TaskSendParams,
    TextPart,
    Part,
)


class PurchasingAgent:
    """The purchasing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: List[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            try:
                card = card_resolver.get_agent_card()
                # The URL accessed here should be the same as the one provided in the agent card
                # However, in this demo we are using the URL provided in the key arguments
                remote_connection = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.remote_agent_connections[card.name] = remote_connection
                self.cards[card.name] = card
            except httpx.ConnectError:
                print(f"ERROR: Failed to get agent card from : {address}")
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = "\n".join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model=LiteLlm(model=OLLAMA_CHAT_MODEL, api_base=OLLAMA_BASE_URL),
            name="purchasing_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This purchasing agent orchestrates the decomposition of the user purchase request into"
                " tasks that can be performed by the seller agents."
            ),
            tools=[
                self.send_task,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_active_agent(context)
        return f"""You are an expert purchasing delegator that can delegate the user product inquiry and purchase request to the
appropriate seller remote agents.

Execution:
- For actionable tasks, you can use `send_task` to assign tasks to remote agents to perform.
- When the remote agent is repeatedly asking for user confirmation, assume that the remote agent doesn't have access to user's conversation context. 
    So improve the task description to include all the necessary information related to that agent
- Never ask user permission when you want to connect with remote agents. If you need to make connection with multiple remote agents, directly
    connect with them without asking user permission or asking user preference
- Always show the detailed response information from the seller agent and propagate it properly to the user. 
- If the remote seller is asking for confirmation, rely the confirmation question to the user if the user haven't do so. 
- If the user already confirmed the related order in the past conversation history, you can confirm on behalf of the user
- Do not give irrelevant context to remote seller agent. For example, ordered pizza item is not relevant for the burger seller agent
- Never ask order confirmation to the remote seller agent 

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

If there is an active agent, send the request to that agent with the update task tool.

Agents:
{self.agents}

Current active seller agent: {current_agent["active_agent"]}
"""

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f"{state['active_agent']}"}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump()}")
            print("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_task(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to remote seller agent

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        if "task_id" in state:
            taskId = state["task_id"]
        else:
            taskId = str(uuid.uuid4())
        sessionId = state["session_id"]
        task: Task
        messageId = ""
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])
            if "message_id" in state["input_message_metadata"]:
                messageId = state["input_message_metadata"]["message_id"]
        if not messageId:
            messageId = str(uuid.uuid4())
        metadata.update(**{"conversation_id": sessionId, "message_id": messageId})
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=sessionId,
            message=Message(
                role="user",
                parts=[TextPart(text=task)],
                metadata=metadata,
            ),
            acceptedOutputModes=["text", "text/plain"],
            # pushNotification=None,
            metadata={"conversation_id": sessionId},
        )
        task = await client.send_task(request, self.task_callback)
        # Assume completion unless a state returns that isn't complete
        state["session_active"] = task.status.state not in [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.FAILED,
            TaskState.UNKNOWN,
        ]
        if task.status.state == TaskState.INPUT_REQUIRED:
            # Force user input back
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.COMPLETED:
            # Reset active agent is task is completed
            state["active_agent"] = "None"

        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(convert_parts(task.status.message.parts, tool_context))
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(convert_parts(artifact.parts, tool_context))
        return response


def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def convert_part(part: Part, tool_context: ToolContext):
    # Currently only support text parts
    if part.type == "text":
        return part.text

    return f"Unknown type: {part.type}"


import os

root_agent = PurchasingAgent(
    remote_agent_addresses=[
        PIZZA_SELLER_AGENT_URL,
        BURGER_SELLER_AGENT_URL,
    ]
).create_agent()

import gradio as gr
from typing import List, Dict, Any
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from typing import AsyncIterator
from google.genai import types
from pprint import pformat
import logging
logging.getLogger('opentelemetry.context').setLevel(50) # Suppress Opentelemetry logs

purchasing_agent = root_agent

APP_NAME = "purchasing_concierge_app"
USER_ID = "default_user"
SESSION_ID = "default_session"
SESSION_SERVICE = InMemorySessionService()
PURCHASING_AGENT_RUNNER = Runner(
    agent=purchasing_agent,  # The agent we want to run
    app_name=APP_NAME,  # Associates runs with our app
    session_service=SESSION_SERVICE,  # Uses our session manager
)
import asyncio

asyncio.run(
    SESSION_SERVICE.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
)


async def get_response_from_agent(
    message: str,
    history: List[Dict[str, Any]],
) -> str:
    """Send the message to the backend and get a response.

    Args:
        message: Text content of the message.
        history: List of previous message dictionaries in the conversation.

    Returns:
        Text response from the backend service.
    """
    # try:
    events_iterator: AsyncIterator[Event] = PURCHASING_AGENT_RUNNER.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    )

    responses = []
    async for event in events_iterator:  # event has type Event
        if event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    formatted_call = f"```python\n{pformat(part.function_call.model_dump(), indent=2, width=80)}\n```"
                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=f"{part.function_call.name}:\n{formatted_call}",
                            metadata={"title": "🛠️ Tool Call"},
                        )
                    )
                elif part.function_response:
                    formatted_response = f"```python\n{pformat(part.function_response.model_dump(), indent=2, width=80)}\n```"

                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=formatted_response,
                            metadata={"title": "⚡ Tool Response"},
                        )
                    )

        # Key Concept: is_final_response() marks the concluding message for the turn
        if event.is_final_response():
            if event.content and event.content.parts:
                # Extract the final answer without the model's reasoning.
                final_response_text = "\n".join(
                    part.text for part in event.content.parts
                    if part.text and not part.thought
                )
            elif event.actions and event.actions.escalate:
                # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            responses.append(
                gr.ChatMessage(role="assistant", content=final_response_text)
            )
            yield responses
            break  # Stop processing events once the final response is found

        yield responses

if __name__ == "__main__":
    import os

    # from jupyter_server.serverapp import list_running_servers
    # base_url = next(list_running_servers())["base_url"].rstrip("/")
    # app_base = f"{base_url}/proxy/8087"

    demo = gr.ChatInterface(
        get_response_from_agent,
        title="Purchasing Concierge",
        description="This assistant can help you to purchase food from remote sellers.",
    )

    demo.launch(
        server_name=os.getenv("CONCIERGE_HOST", "0.0.0.0"),
        server_port=int(os.getenv("CONCIERGE_PORT", "8087")),
        share=False,
        inline=False,
        # root_path=app_base
    )
    # display(IFrame(src=app_base + "/", width="100%", height=600))
    print("浏览器打开 http://127.0.0.1:8087  (需先开隧道)")
    # source /root/tmp/agents-201/.venv/bin/activate