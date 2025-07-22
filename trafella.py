"""
Trafella - Your AI Travel Companion
----------------------------------
Trafella is a web-based travel planning application using Streamlit.
It leverages AI to provide flight options, destination information, personalized
itineraries, and hotel/restaurant recommendations.

Key functionalities:
- User input for travel details (destination, dates, preferences).
- API integration with SerpAPI for real-time flight data.
- AI-powered planning using Google's Gemini model.
- Local translation for multilingual support.
- Secure handling of API keys using Streamlit's session state.
"""

# ==============================================================================
# 1. LIBRARY IMPORTS
# ==============================================================================
# Import necessary libraries
import streamlit as st  # Main library for building the web interface
import json  # For working with JSON data structures
import os  # For interacting with the operating system (environment variables, etc.)
import logging  # For application logging and debugging
from datetime import datetime  # For handling dates and times
from serpapi.google_search import GoogleSearch  # For fetching flight data from SerpAPI
from agno.agent import Agent  # For creating AI agents
from agno.tools.serpapi import SerpApiTools  # SerpAPI tools for AI agents
from agno.models.google import Gemini  # Google's Gemini AI model integration
from transformers import MarianMTModel, MarianTokenizer  # For local text translation
import torch  # PyTorch for machine learning operations

# ==============================================================================
# 2. INITIAL SETUP AND CONFIGURATION
# ==============================================================================

# --- Logging Configuration ---
# Sets up a logging system to record application events. This is crucial for
# debugging and monitoring the application's behavior. Logs are saved to a
# file named with the current date.
log_filename = datetime.now().strftime("%Y%m%d") + "-trafella.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Trafella")
logger.info("Application started")

# --- API Key Management ---
# This section handles the API keys required for SerpAPI (flight data) and
# Google (AI models). It securely prompts the user for their keys and stores
# them in Streamlit's session state, which is temporary and browser-based.
# The application will not proceed until the keys are provided.
if "SERPAPI_KEY" not in st.session_state or "GOOGLE_API_KEY" not in st.session_state:
    with st.sidebar.expander("🔑 API Keys Setup", expanded=True):
        st.markdown("### 🔑 API Keys Required")
        st.markdown("Please enter your API keys to continue:")

        user_serpapi = st.text_input(
            "SerpAPI Key",
            type="password",
            help="Get your key from https://serpapi.com/manage-api-key",
            key="serpapi_input"
        )
        user_google = st.text_input(
            "Google API Key",
            type="password",
            help="Get your key from https://aistudio.google.com/app/u/1/apikey",
            key="google_input"
        )


        if st.button("💾 Save API Keys"):
            if user_serpapi and user_google:
                st.session_state["SERPAPI_KEY"] = user_serpapi
                st.session_state["GOOGLE_API_KEY"] = user_google
                st.success("API keys saved for this session!")
                st.rerun()
            else:
                st.error("Please enter both API keys")

        st.markdown("---")
        st.markdown("*Your keys are only stored in your browser for this session and never saved on the server.*")

    st.warning("Please enter your API keys in the sidebar to continue.")
    st.stop()

# --- Load API Keys from Session State ---
# Once the user provides the keys, they are loaded from the session state into
# variables for use throughout the application. The Google API key is also set
# as an environment variable, as required by the Gemini model's library.
SERPAPI_KEY = st.session_state["SERPAPI_KEY"]
GOOGLE_API_KEY = st.session_state["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # For Gemini agent

# Log API key status (without exposing actual keys)
logger.info("API keys set in session_state")

# ==============================================================================
# 3. TRANSLATION FUNCTIONALITY
# ==============================================================================

# --- Translation Model Loading ---
# This function loads a pre-trained translation model from Hugging Face.
# The `@st.cache_resource` decorator ensures the model is loaded only once
# and cached, which significantly improves performance by avoiding repeated
# downloads and loading on each script rerun.
@st.cache_resource
def get_translation_model_and_tokenizer():
    """
    Loads and caches the translation model and tokenizer.
    """
    try:
        logger.info("Loading translation model and tokenizer...")
        model_name = 'Helsinki-NLP/opus-mt-en-id'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        logger.info("Translation model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load translation model: {e}")
        logger.error(f"Failed to load translation model: {e}")
        return None, None

# --- Text Translation Function ---
# This function takes English text as input and translates it into Bahasa
# Indonesia. It handles the text in smaller chunks (sentences) to avoid
# overwhelming the model and to maintain context during translation.
def translate_text(text_to_translate, chunk_size=512):
    """
    Translates text to Indonesian using a local Hugging Face model.
    """
    model, tokenizer = get_translation_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "Error: Translation model not available."

    if not text_to_translate or not text_to_translate.strip():
        logger.warning("Input text for translation is empty or just whitespace.")
        return ""

    try:
        logger.info(f"Translating text locally: {text_to_translate[:50]}...")
        # Split the text into chunks based on sentences to maintain context
        sentences = text_to_translate.split('. ')
        translated_chunks = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)
            with torch.no_grad():
                translated_tokens = model.generate(**inputs)
            
            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
            if decoded:
                translated_chunks.append(decoded[0])
            else:
                logger.warning(f"Translation of chunk returned empty result: '{sentence[:50]}...'")

        translated_text = '. '.join(translated_chunks)
        logger.info("Local translation successful.")
        return translated_text

    except Exception as e:
        logger.error(f"An error occurred during local translation: {e}")
        return f"Error: Could not translate text locally. {e}"


# ==============================================================================
# 4. USER INTERFACE (UI) SETUP
# ==============================================================================

# --- Page and Style Configuration ---
# This section configures the main Streamlit page, setting the title and layout.
# It also injects custom CSS to style the title, subtitle, and other elements
# for a more polished look.
st.set_page_config(page_title="🌍 Trafella", layout="wide")
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff5733;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .stSlider > div {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Main Title and Subtitle ---
# Displays the main header and a brief description of the application at the
# top of the page.
st.markdown('<h1 class="title">✈️ Trafella - Your AI Travel Companion</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plan your ideal holiday with AI assistance! Receive customized suggestions for flights, accommodations, and activities.</p>', unsafe_allow_html=True)

# --- Main User Input Section ---
# This is where the user enters the core details of their trip, such as
# departure/destination cities, travel duration, theme, and activity
# preferences. These inputs are the foundation for the AI's planning process.
st.markdown("### 🗺️ Enter your travel destinations:")
source = st.text_input("🛫 Departure City (IATA Code example: CGK for Jakarta):", "CGK")  # Example: CGK for Jakarta
destination = st.text_input("🛬 Destination (IATA Code example: KUL for Kuala Lumpur):", "KUL")  # Example: KUL for Kuala Lumpur

st.markdown("### 📅 Customize Your Travel")
num_days = st.slider("🕒 Travel Duration (days):", 1, 14, 5)
travel_theme = st.selectbox(
    "🎭 Select Your Travel Theme:",
    ["💑 Couple Getaway", "👨‍👩‍👧‍👦 Family Vacation", "🏔️ Adventure Trip", "🧳 Solo Exploration"]
)

activity_preferences = st.text_area(
    "🌍 What activities do you enjoy? (e.g., relaxing on the beach, exploring historical sites, nightlife, adventure)",
    "Relaxing on the beach, exploring historical sites"
)

departure_date = st.date_input("Departure Date")
return_date = st.date_input("Return Date")

# Add language toggle option
language = st.selectbox("🌐 Language", ["English", "Bahasa Indonesia"])


# --- Sidebar for Additional Preferences ---
# The sidebar contains more granular options for personalizing the trip,
# including budget, flight class, and preferred hotel rating. This allows
# for more tailored recommendations.
st.sidebar.title("🌎 Travel Assistant")
st.sidebar.subheader("Personalize Your Trip")

# Travel Preferences
budget = st.sidebar.radio("💰 Budget Preference:", ["Economy", "Standard", "Luxury"])
flight_class = st.sidebar.radio("✈️ Flight Class:", ["Economy", "Business", "First Class"])
hotel_rating = st.sidebar.selectbox("🏨 Preferred Hotel Rating:", ["Any", "3⭐", "4⭐", "5⭐"])

params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "IDR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }

# ==============================================================================
# 5. DATA FETCHING AND PROCESSING FUNCTIONS
# ==============================================================================

# --- Datetime Formatting Function ---
# A helper function to convert date strings from the API into a more
# readable format (e.g., "Jul-21, 2025 | 6:20 PM").
# Function to format datetime
def format_datetime(iso_string):
    try:
        logger.info(f"Formatting datetime: {iso_string}")
        dt = datetime.strptime(iso_string, "%Y-%m-%d %H:%M")
        return dt.strftime("%b-%d, %Y | %I:%M %p")  # Example: Jul-21, 2025 | 6:20 PM
    except ValueError as e:
        logger.error(f"Error formatting datetime: {e}")
        return "N/A"

# --- Flight Data Fetching Function ---
# This function calls the SerpAPI Google Flights endpoint with the user's
# travel details to get real-time flight information.
# Function to fetch flight data
def fetch_flights(source, destination, departure_date, return_date):
    logger.info(f"Fetching flights for {source} to {destination} from {departure_date} to {return_date}")
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "IDR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    logger.info(f"Flight data fetched successfully: {results}")
    return results

# --- Flight Data Extraction Function ---
# After fetching the flight data, this function processes the results to
# find and return the top 3 cheapest flight options.
# Function to extract top 3 cheapest flights
def extract_cheapest_flights(flight_data):
    logger.info("Extracting cheapest flights")
    best_flights = flight_data.get("best_flights", [])
    sorted_flights = sorted(best_flights, key=lambda x: x.get("price", float("inf")))[:3]  # Get top 3 cheapest
    logger.info(f"Top 3 cheapest flights extracted: {len(sorted_flights)} flights found")
    return sorted_flights

# ==============================================================================
# 6. AI AGENT CONFIGURATION
# ==============================================================================
# This section defines the specialized AI agents that will handle different
# parts of the travel planning task. Each agent has a specific role and set
# of instructions.

# --- Researcher Agent ---
# This agent is responsible for gathering general information about the
# destination, including attractions, culture, and safety tips.
researcher = Agent(
    name="Researcher",
    instructions=[
        "Identify the travel destination specified by the user.",
        "Gather detailed information on the destination, including climate, culture, and safety tips.",
        "Find popular attractions, landmarks, and must-visit places.",
        "Search for activities that match the user’s interests and travel style.",
        "Prioritize information from reliable sources and official travel guides.",
        "Provide well-structured summaries with key insights and recommendations."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# --- Planner Agent ---
# The planner agent takes all the gathered information (research, flights,
# hotels) and synthesizes it into a structured, day-by-day itinerary.
planner = Agent(
    name="Planner",
    instructions=[
        "Gather details about the user's travel preferences and budget.",
        "Create a detailed itinerary with scheduled activities and estimated costs.",
        "Ensure the itinerary includes transportation options and travel time estimates.",
        "Optimize the schedule for convenience and enjoyment.",
        "Present the itinerary in a structured format."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    add_datetime_to_instructions=True,
)

# --- Hotel & Restaurant Finder Agent ---
# This agent specializes in finding accommodation and dining options that
# match the user's preferences for budget, rating, and location.
hotel_restaurant_finder = Agent(
    name="Hotel & Restaurant Finder",
    instructions=[
        "Identify key locations in the user's travel itinerary.",
        "Search for highly rated hotels near those locations.",
        "Search for top-rated restaurants based on cuisine preferences and proximity.",
        "Prioritize results based on user preferences, ratings, and availability.",
        "Provide direct booking links or reservation options where possible."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# ==============================================================================
# 7. MAIN APPLICATION LOGIC
# ==============================================================================
# This section contains the main logic of the application, including fetching
# flight data, running the AI agents, and displaying the results.

# --- Travel Plan Generation Trigger ---
# This is the main execution block of the application. It runs when the user
# clicks the "Generate Travel Plan" button.
if st.button("🚀 Generate Travel Plan"):
    with st.spinner("✈️ Fetching best flight options..."):
        logger.info("Fetching flights")
        flight_data = fetch_flights(source, destination, departure_date, return_date)
        logger.info(f"Flight data fetched successfully: {flight_data}")
        cheapest_flights = extract_cheapest_flights(flight_data)
        logger.info(f"Cheapest flights extracted: {cheapest_flights}")

    # --- AI-Powered Data Processing ---
    # This block orchestrates the AI agents. It shows spinners to the user
    # while the agents perform their tasks: researching the destination,
    # finding hotels/restaurants, and creating the final itinerary.
    with st.spinner("🔍 Researching best attractions & activities..."):
        logger.info("Researching attractions and activities")
        research_prompt = (
            f"Research the best attractions and activities in {destination} for a {num_days}-day {travel_theme.lower()} trip. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. "
            f"Hotel Rating: {hotel_rating}."
        )
        logger.info(f"Research prompt: {research_prompt}")
        research_results = researcher.run(research_prompt, stream=False)
        logger.info(f"Research results: {research_results}")

    with st.spinner("🏨 Searching for hotels & restaurants..."):
        logger.info("Searching for hotels and restaurants")
        hotel_restaurant_prompt = (
            f"Find the best hotels and restaurants near popular attractions in {destination} for a {travel_theme.lower()} trip. "
            f"Budget: {budget}. Hotel Rating: {hotel_rating}. Preferred activities: {activity_preferences}."
        )
        logger.info(f"Hotel and restaurant prompt: {hotel_restaurant_prompt}")
        hotel_restaurant_results = hotel_restaurant_finder.run(hotel_restaurant_prompt, stream=False)
        logger.info(f"Hotel and restaurant results: {hotel_restaurant_results}")

    with st.spinner("🗺️ Creating your personalized itinerary..."):
        logger.info("Creating itinerary")
        planning_prompt = (
            f"Based on the following data, create a {num_days}-day itinerary for a {travel_theme.lower()} trip to {destination}. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. Hotel Rating: {hotel_rating}. "
            f"Research: {research_results.content}. "
            f"Flights: {json.dumps(cheapest_flights)}. Hotels & Restaurants: {hotel_restaurant_results.content}."
        )
        logger.info(f"Planning prompt: {planning_prompt}")
        itinerary = planner.run(planning_prompt, stream=False)
        logger.info(f"Itinerary: {itinerary}")

        # ==============================================================================
    # 8. RESULTS DISPLAY
    # ==============================================================================
    # After the AI has finished processing, this section displays the results
    # to the user in a structured and visually appealing format.

    # --- Display Flight Options ---
    # Renders the top 3 cheapest flight options in a card layout, including
    # airline logos, prices, and direct booking links.
    st.subheader("✈️ Cheapest Flight Options")
    if cheapest_flights:
        logger.info("Displaying cheapest flight options")
        cols = st.columns(len(cheapest_flights))
        for idx, flight in enumerate(cheapest_flights):
            with cols[idx]:
                logger.info(f"Displaying flight {idx + 1}")
                airline_logo = flight.get("airline_logo", "")
                airline_name = flight.get("airline", "Unknown Airline")
                price = flight.get("price", "Not Available")
                total_duration = flight.get("total_duration", "N/A")
                
                flights_info = flight.get("flights", [{}])
                departure = flights_info[0].get("departure_airport", {})
                arrival = flights_info[-1].get("arrival_airport", {})
                airline_name = flights_info[0].get("airline", "Unknown Airline") 
                
                departure_time = format_datetime(departure.get("time", "N/A"))
                arrival_time = format_datetime(arrival.get("time", "N/A"))
                
                departure_token = flight.get("departure_token", "")

                if departure_token:
                    logger.info("Fetching booking options")
                    params_with_token = {
                        **params,
                        "departure_token": departure_token  # Add the token here
                    }
                    search_with_token = GoogleSearch(params_with_token)
                    results_with_booking = search_with_token.get_dict()

                    booking_options = results_with_booking['best_flights'][idx]['booking_token']

                booking_link = f"https://www.google.com/travel/flights?tfs="+booking_options if booking_options else "#"
                logger.info(f"Booking link: {booking_link}")
                # Flight card layout
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center;
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        margin-bottom: 20px;
                    ">
                        <img src="{airline_logo}" width="100" alt="Flight Logo" />
                        <h3 style="margin: 10px 0;">{airline_name}</h3>
                        <p><strong>Departure:</strong> {departure_time}</p>
                        <p><strong>Arrival:</strong> {arrival_time}</p>
                        <p><strong>Duration:</strong> {total_duration} min</p>
                        <h2 style="color: #008000;">💰 {price}</h2>
                        <a href="{booking_link}" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #fff;
                            background-color: #007bff;
                            text-decoration: none;
                            border-radius: 5px;
                            margin-top: 10px;
                        ">🔗 Book Now</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        logger.info("No flight data available")
        st.warning("⚠️ No flight data available.")

    # --- Display Personalized Itinerary ---
    # Presents the final, detailed travel plan created by the Planner agent.
    # The content is translated if needed.
    st.subheader("🗺️ Your Personalized Itinerary")
    logger.info("Displaying itinerary")
    if language == "Bahasa Indonesia":
        logger.info(f"Itinerary before translation: {itinerary.content}")
        translated_content = translate_text(itinerary.content)
        logger.info(f"Translated itinerary: {translated_content}")
        st.write(translated_content)
    else:
        logger.info(f"Itinerary: {itinerary.content}")
        st.write(itinerary.content)

    st.success("✅ Travel plan generated successfully!")
