import os

from typing import Literal
from pydantic import BaseModel
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

from crewai.tools import tool

from crewai import Agent as crewagent
from crewai import Crew as crewcrew
from crewai import LLM as crewllm
from crewai import Task as crewtask
from crewai import Process as crewprocess

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


@tool("create_burger_order")
def create_burger_order(order_items: list[OrderItem]) -> str:
    """
    Creates a new burger order with the given order items.

    Args:
        order_items: List of items with name (str), quantity (int), and price
            (int, unit price in IDR).

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

class BurgerSellerAgent:
    TaskInstruction = """
# INSTRUCTIONS

You are a specialized assistant for a burger store.
Your sole purpose is to answer questions about what is available on burger menu and price also handle order creation.
If the user asks about anything other than burger menu or order creation, politely state that you cannot help with that topic and can only assist with burger menu and order creation.
Do not attempt to answer unrelated questions or use tools for other purposes.

# CONTEXT

Received user query: {user_prompt}
Session ID: {session_id}

Provided below is the available burger menu and it's related price:
- Classic Cheeseburger: IDR 85K
- Double Cheeseburger: IDR 110K
- Spicy Chicken Burger: IDR 80K
- Spicy Cajun Burger: IDR 85K

# RULES

- If user want to do something, you will be following this order:
    1. Always ensure the user already confirmed the order and total price. This confirmation may already given in the user query. An intent to order is not confirmation of the total price.
    2. Use `create_burger_order` tool to create the order
    3. Finally, always provide response to the user about the detailed ordered items, price breakdown and total, and order ID
    
- Set response status to input_required if asking for user order confirmation.
- Set response status to error if there is an error while processing the request.
- Set response status to completed if the request is complete.
- DO NOT make up menu or price, Always rely on the provided menu given to you as context.
"""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def invoke(self, query, sessionId) -> str:
        burger_agent = crewagent(
            role="Burger Seller Agent",
            goal=(
                "Help user to understand what is available on burger menu and price also handle order creation."
            ),
            backstory=("You are an expert and helpful burger seller agent."),
            verbose=False,
            allow_delegation=False,
            tools=[create_burger_order],
            llm=crewllm(
                model=os.getenv("VLLM_MODEL", "qwen35"),
                provider="openai",
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8088/v1"),
                api_key="EMPTY"
                )
        )

        agent_task = crewtask(
            description=self.TaskInstruction,
            output_pydantic=ResponseFormat,
            agent=burger_agent,
            expected_output=(
                "A JSON object with 'status' and 'message' fields."
                "Set response status to input_required if asking for user order confirmation."
                "Set response status to error if there is an error while processing the request."
                "Set response status to completed if the request is complete."
            ),
        )

        crew = crewcrew(
            tasks=[agent_task],
            agents=[burger_agent],
            verbose=False,
            process=crewprocess.sequential,
        )

        inputs = {"user_prompt": query, "session_id": sessionId}
        response = crew.kickoff(inputs)
        return self.get_agent_response(response)

    def get_agent_response(self, response):
        response_object = response.pydantic
        if response_object and isinstance(response_object, ResponseFormat):
            if response_object.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response_object.message,
                }
            elif response_object.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response_object.message,
                }
            elif response_object.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": response_object.message,
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
import logging
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUTH_USERNAME=os.getenv("BURGER_AGENT_USER", "burgeruser123")
AUTH_PASSWORD=os.getenv("BURGER_AGENT_PASSWORD", "burgerpass123")

def main(host, port):   

    """Starts the Burger Seller Agent server."""
    try:
        capabilities = AgentCapabilities(pushNotifications=True)
        skill = AgentSkill(
            id="create_burger_order",
            name="Burger Order Creation Tool",
            description="Helps with creating burger orders",
            tags=["burger order creation"],
            examples=["I want to order 2 classic cheeseburgers"],
        )
        agent_card = AgentCard(
            name="burger_seller_agent",
            description="Helps with creating burger orders",
            # The URL provided here is for the sake of demo,
            # in production you should use a proper domain name
            url=f"http://{host}:{port}/",
            version="1.0.0",
            authentication=AgentAuthentication(schemes=["Basic"]),
            defaultInputModes=BurgerSellerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=BurgerSellerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )


        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=BurgerSellerAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host=host,
            port=port,
            auth_username=AUTH_USERNAME,
            auth_password=AUTH_PASSWORD,
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

# --- Global variable to hold the server thread reference ---
# This allows you to stop it later from another cell if needed
global server_thread
server_thread = None

# --- Main execution in the Jupyter cell ---
if server_thread is not None and server_thread.is_alive():
    print("Server is already running.")
else:
    # Define host and port (from .env)
    server_host = os.getenv("BURGER_AGENT_HOST", "0.0.0.0")
    server_port = int(os.getenv("BURGER_AGENT_PORT", "10001"))

    # Create and start the thread
    server_thread = threading.Thread(target=main, args=(server_host, server_port))
    server_thread.daemon = True # Allows the main program to exit even if the thread is still running
    server_thread.start()

    print(f"Server thread started. Waiting a moment for server to initialize on http://{server_host}:{server_port}")
    time.sleep(5) # Give it a few seconds to boot up
    server_thread.join()            # ← 加这一行