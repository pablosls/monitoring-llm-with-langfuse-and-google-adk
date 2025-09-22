import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

from langfuse import get_client
 
langfuse = get_client()


from openinference.instrumentation.google_adk import GoogleADKInstrumentor
 
GoogleADKInstrumentor().instrument()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model=LiteLlm("ollama_chat/gpt-oss:20b"),
    description=(
        "Agent to answer questions about the time and weather in a city"
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)


async def main():
    
    APP_NAME = "hello_app"
    USER_ID = "demo-user"
    SESSION_ID = "demo-session"
    
    session_service = InMemorySessionService()
    # create_session is async → await it in notebooks
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    
    user_msg = types.Content(role="user", parts=[types.Part(text="hi")])
    for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            print(event.content.parts[0].text)
        
if __name__ == "__main__":
    asyncio.run(main())