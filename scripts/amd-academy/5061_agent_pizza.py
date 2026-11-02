from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing import Literal
from pydantic import BaseModel
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

memory = MemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("__main__", "ResponseFormat")
        ]
    )
)
# memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

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


@tool
def create_pizza_order(order_items: list[OrderItem]) -> str:
    """
    Creates a new pizza order with the given order items.

    Args:
        order_items: List of order items to be added to the order.

    Returns:
        str: A message indicating that the order has been created.
    """
    try:
        order_id = str(uuid.uuid4())
        order = Order(order_id=order_id, status="created", order_items=order_items)
        print("===")
        print(f"order created: {order}")
        print("===")
    except Exception as e:
        print(f"Error creating order: {e}")
        return f"Error creating order: {e}"
    return f"Order {order.model_dump()} has been created"

class PizzaSellerAgent:
    SYSTEM_INSTRUCTION = """
# INSTRUCTIONS

You are a specialized assistant for a pizza store.
Your sole purpose is to answer questions about what is available on pizza menu and price also handle order creation.
If the user asks about anything other than pizza menu or order creation, politely state that you cannot help with that topic and can only assist with pizza menu and order creation.
Do not attempt to answer unrelated questions or use tools for other purposes.

# CONTEXT

Provided below is the available pizza menu and it's related price:
- Margherita Pizza: IDR 100K
- Pepperoni Pizza: IDR 140K
- Hawaiian Pizza: IDR 110K
- Veggie Pizza: IDR 100K
- BBQ Chicken Pizza: IDR 130K

# RULES

- If user want to do something, you will be following this order:
    1. Always ensure the user already confirmed the order and total price. This confirmation may already given in the user query. An intent to order is not confirmation of the total price.
    2. Use `create_pizza_order` tool to create the order
    3. Finally, always provide response to the user about the detailed ordered items, price breakdown and total, and order ID

- Set response status to input_required if asking for user order confirmation.
- Set response status to error if there is an error while processing the request.
- Set response status to completed if the request is complete.
- DO NOT make up menu or price, Always rely on the provided menu given to you as context.
"""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):

        self.model = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen35"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        self.tools = [create_pizza_order]
        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
        )

    def invoke(self, query, sessionId) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        self.graph.invoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = self.model.with_structured_output(ResponseFormat).invoke([
            ("system", "Copy the provided reply verbatim into message. "
             "Status describes the provided reply: input_required if it asks the user "
             "a question or requests confirmation; error if it reports a failure; completed otherwise."),
            ("user", current_state.values["messages"][-1].content),
        ], reasoning=False)
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

from utils.server import A2AServer
from utils.a2a_types import AgentCard, AgentCapabilities, AgentSkill, AgentAuthentication
from utils.push_notification_auth import PushNotificationSenderAuth
from utils.task_manager import AgentTaskManager
import click
import logging
import os
import threading
import time
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(host, port):
    """Starts the Pizza Seller Agent server."""
    try:
        capabilities = AgentCapabilities(pushNotifications=True)
        skill = AgentSkill(
            id="create_pizza_order",
            name="Pizza Order Creation Tool",
            description="Helps with creating pizza orders",
            tags=["pizza order creation"],
            examples=["I want to order 2 pepperoni pizzas"],
        )
        agent_card = AgentCard(
            name="pizza_seller_agent",
            description="Helps with creating pizza orders",
            # The URL provided here is for the sake of demo,
            # in production you should use a proper domain name
            url=f"http://{host}:{port}/",
            version="1.0.0",
            authentication=AgentAuthentication(schemes=["Bearer"]),
            defaultInputModes=PizzaSellerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PizzaSellerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=PizzaSellerAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host=host,
            port=port,
            api_key=os.getenv("PIZZA_AGENT_AUTH", "pizza123"),
        )

        server.app.add_route(
            "/.well-known/jwks.json",
            notification_sender_auth.handle_jwks_endpoint,
            methods=["GET"],
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)

global server_thread
server_thread = None

# --- Main execution in the Jupyter cell ---
if server_thread is not None and server_thread.is_alive():
    print("Server is already running.")
else:
    # Define host and port (from .env)
    server_host = os.getenv("PIZZA_AGENT_HOST", "0.0.0.0")
    server_port = int(os.getenv("PIZZA_AGENT_PORT", "10000"))

    # Create and start the thread
    server_thread = threading.Thread(target=main, args=(server_host, server_port))
    server_thread.daemon = True # Allows the main program to exit even if the thread is still running
    server_thread.start()

    print(f"Server thread started. Waiting a moment for server to initialize on http://{server_host}:{server_port}")
    time.sleep(5) # Give it a few seconds to boot up
    server_thread.join()            # ← 加这一行
    # source /root/tmp/agents-201/.venv/bin/activate